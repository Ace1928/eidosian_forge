import asyncio
import inspect
import json
import logging
import pickle
import socket
from typing import Any, List, Optional, Type
import starlette
from fastapi.encoders import jsonable_encoder
from starlette.types import ASGIApp, Message, Receive, Scope, Send
from uvicorn.config import Config
from uvicorn.lifespan.on import LifespanOn
from ray._private.pydantic_compat import IS_PYDANTIC_2
from ray.actor import ActorHandle
from ray.serve._private.constants import SERVE_LOGGER_NAME
from ray.serve._private.utils import serve_encoders
from ray.serve.exceptions import RayServeException
def make_fastapi_class_based_view(fastapi_app, cls: Type) -> None:
    """Transform the `cls`'s methods and class annotations to FastAPI routes.

    Modified from
    https://github.com/dmontagu/fastapi-utils/blob/master/fastapi_utils/cbv.py

    Usage:
    >>> from fastapi import FastAPI
    >>> app = FastAPI() # doctest: +SKIP
    >>> class A: # doctest: +SKIP
    ...     @app.route("/{i}") # doctest: +SKIP
    ...     def func(self, i: int) -> str: # doctest: +SKIP
    ...         return self.dep + i # doctest: +SKIP
    >>> # just running the app won't work, here.
    >>> make_fastapi_class_based_view(app, A) # doctest: +SKIP
    >>> # now app can be run properly
    """
    from fastapi import APIRouter, Depends
    from fastapi.routing import APIRoute, APIWebSocketRoute

    def get_current_servable_instance():
        from ray import serve
        return serve.get_replica_context().servable_object
    class_method_routes = [route for route in fastapi_app.routes if isinstance(route, (APIRoute, APIWebSocketRoute)) and cls.__qualname__ in route.endpoint.__qualname__]
    new_router = APIRouter()
    for route in class_method_routes:
        fastapi_app.routes.remove(route)
        old_endpoint = route.endpoint
        old_signature = inspect.signature(old_endpoint)
        old_parameters = list(old_signature.parameters.values())
        if len(old_parameters) == 0:
            raise RayServeException('Methods in FastAPI class-based view must have ``self`` as their first argument.')
        old_self_parameter = old_parameters[0]
        new_self_parameter = old_self_parameter.replace(default=Depends(get_current_servable_instance))
        new_parameters = [new_self_parameter] + [parameter.replace(kind=inspect.Parameter.KEYWORD_ONLY) for parameter in old_parameters[1:]]
        new_signature = old_signature.replace(parameters=new_parameters)
        setattr(route.endpoint, '__signature__', new_signature)
        setattr(route.endpoint, '_serve_cls', cls)
        new_router.routes.append(route)
    fastapi_app.include_router(new_router)
    routes_to_remove = list()
    for route in fastapi_app.routes:
        if not isinstance(route, (APIRoute, APIWebSocketRoute)):
            continue
        if not IS_PYDANTIC_2 and isinstance(route, APIRoute) and route.response_model:
            route.secure_cloned_response_field.outer_type_ = route.response_field.outer_type_
        serve_cls = getattr(route.endpoint, '_serve_cls', None)
        if serve_cls is not None and serve_cls != cls:
            routes_to_remove.append(route)
    fastapi_app.routes[:] = [r for r in fastapi_app.routes if r not in routes_to_remove]