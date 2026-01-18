import asyncio
import collections
import functools
import inspect
import json
import logging
import os
import time
import traceback
from collections import namedtuple
from typing import Any, Callable
from aiohttp.web import Response
import ray
import ray.dashboard.consts as dashboard_consts
from ray._private.ray_constants import RAY_INTERNAL_DASHBOARD_NAMESPACE, env_bool
from ray.dashboard.optional_deps import PathLike, RouteDef, aiohttp, hdrs
from ray.dashboard.utils import CustomEncoder, to_google_style
def method_route_table_factory():

    class MethodRouteTable:
        """A helper class to bind http route to class method."""
        _bind_map = collections.defaultdict(dict)
        _routes = aiohttp.web.RouteTableDef()

        class _BindInfo:

            def __init__(self, filename, lineno, instance):
                self.filename = filename
                self.lineno = lineno
                self.instance = instance

        @classmethod
        def routes(cls):
            return cls._routes

        @classmethod
        def bound_routes(cls):
            bound_items = []
            for r in cls._routes._items:
                if isinstance(r, RouteDef):
                    route_method = getattr(r.handler, '__route_method__')
                    route_path = getattr(r.handler, '__route_path__')
                    instance = cls._bind_map[route_method][route_path].instance
                    if instance is not None:
                        bound_items.append(r)
                else:
                    bound_items.append(r)
            routes = aiohttp.web.RouteTableDef()
            routes._items = bound_items
            return routes

        @classmethod
        def _register_route(cls, method, path, **kwargs):

            def _wrapper(handler):
                if path in cls._bind_map[method]:
                    bind_info = cls._bind_map[method][path]
                    raise Exception(f'Duplicated route path: {path}, previous one registered at {bind_info.filename}:{bind_info.lineno}')
                bind_info = cls._BindInfo(handler.__code__.co_filename, handler.__code__.co_firstlineno, None)

                @functools.wraps(handler)
                async def _handler_route(*args) -> aiohttp.web.Response:
                    try:
                        req = args[-1]
                        return await handler(bind_info.instance, req)
                    except Exception:
                        logger.exception('Handle %s %s failed.', method, path)
                        return rest_response(success=False, message=traceback.format_exc())
                cls._bind_map[method][path] = bind_info
                _handler_route.__route_method__ = method
                _handler_route.__route_path__ = path
                return cls._routes.route(method, path, **kwargs)(_handler_route)
            return _wrapper

        @classmethod
        def head(cls, path, **kwargs):
            return cls._register_route(hdrs.METH_HEAD, path, **kwargs)

        @classmethod
        def get(cls, path, **kwargs):
            return cls._register_route(hdrs.METH_GET, path, **kwargs)

        @classmethod
        def post(cls, path, **kwargs):
            return cls._register_route(hdrs.METH_POST, path, **kwargs)

        @classmethod
        def put(cls, path, **kwargs):
            return cls._register_route(hdrs.METH_PUT, path, **kwargs)

        @classmethod
        def patch(cls, path, **kwargs):
            return cls._register_route(hdrs.METH_PATCH, path, **kwargs)

        @classmethod
        def delete(cls, path, **kwargs):
            return cls._register_route(hdrs.METH_DELETE, path, **kwargs)

        @classmethod
        def view(cls, path, **kwargs):
            return cls._register_route(hdrs.METH_ANY, path, **kwargs)

        @classmethod
        def static(cls, prefix: str, path: PathLike, **kwargs: Any) -> None:
            cls._routes.static(prefix, path, **kwargs)

        @classmethod
        def bind(cls, instance):

            def predicate(o):
                if inspect.ismethod(o):
                    return hasattr(o, '__route_method__') and hasattr(o, '__route_path__')
                return False
            handler_routes = inspect.getmembers(instance, predicate)
            for _, h in handler_routes:
                cls._bind_map[h.__func__.__route_method__][h.__func__.__route_path__].instance = instance
    return MethodRouteTable