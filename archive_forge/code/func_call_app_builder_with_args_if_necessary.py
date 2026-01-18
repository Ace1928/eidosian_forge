import inspect
import logging
from types import FunctionType
from typing import Any, Dict, Tuple, Union
import ray
from ray._private.pydantic_compat import is_subclass_of_base_model
from ray._private.resource_spec import HEAD_NODE_RESOURCE_NAME
from ray._private.usage import usage_lib
from ray.actor import ActorHandle
from ray.serve._private.client import ServeControllerClient
from ray.serve._private.constants import (
from ray.serve._private.controller import ServeController
from ray.serve.config import HTTPOptions, gRPCOptions
from ray.serve.context import _get_global_client, _set_global_client
from ray.serve.deployment import Application, Deployment
from ray.serve.exceptions import RayServeException
from ray.serve.schema import LoggingConfig
def call_app_builder_with_args_if_necessary(builder: Union[Application, FunctionType], args: Dict[str, Any]) -> Application:
    """Builds a Serve application from an application builder function.

    If a pre-built application is passed, this is a no-op.

    Else, we validate the signature of the builder, convert the args dictionary to
    the user-annotated Pydantic model if provided, and call the builder function.

    The output of the function is returned (must be an Application).
    """
    if isinstance(builder, Application):
        if len(args) > 0:
            raise ValueError('Arguments can only be passed to an application builder function, not an already built application.')
        return builder
    elif not isinstance(builder, FunctionType):
        raise TypeError(f'Expected a built Serve application or an application builder function but got: {type(builder)}.')
    signature = inspect.signature(builder)
    if len(signature.parameters) != 1:
        raise TypeError('Application builder functions should take exactly one parameter, a dictionary containing the passed arguments.')
    param = signature.parameters[list(signature.parameters.keys())[0]]
    if inspect.isclass(param.annotation) and is_subclass_of_base_model(param.annotation):
        args = param.annotation.parse_obj(args)
    app = builder(args)
    if not isinstance(app, Application):
        raise TypeError(f'Application builder functions must return an `Application` returned `from `Deployment.bind()`, but got: {type(app)}.')
    return app