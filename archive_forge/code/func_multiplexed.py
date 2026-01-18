import collections
import inspect
import logging
import warnings
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Union
from fastapi import APIRouter, FastAPI
import ray
from ray import cloudpickle
from ray._private.serialization import pickle_dumps
from ray.dag import DAGNode
from ray.serve._private.config import DeploymentConfig, ReplicaConfig
from ray.serve._private.constants import (
from ray.serve._private.deployment_graph_build import build as pipeline_build
from ray.serve._private.deployment_graph_build import (
from ray.serve._private.http_util import (
from ray.serve._private.usage import ServeUsageTag
from ray.serve._private.utils import (
from ray.serve.config import (
from ray.serve.context import (
from ray.serve.deployment import Application, Deployment
from ray.serve.exceptions import RayServeException
from ray.serve.handle import DeploymentHandle
from ray.serve.multiplex import _ModelMultiplexWrapper
from ray.serve.schema import LoggingConfig, ServeInstanceDetails, ServeStatus
from ray.util.annotations import Deprecated, DeveloperAPI, PublicAPI
from ray.serve._private import api as _private_api  # isort:skip
@PublicAPI(stability='beta')
def multiplexed(func: Optional[Callable[..., Any]]=None, max_num_models_per_replica: int=3):
    """Wrap a callable or method used to load multiplexed models in a replica.

    The function can be standalone function or a method of a class. The
    function must have exactly one argument, the model id of type `str` for the
    model to be loaded.

    It is required to define the function with `async def` and the function must be
    an async function. It is recommended to define coroutines for long running
    IO tasks in the function to avoid blocking the event loop.

    The multiplexed function is called to load a model with the given model ID when
    necessary.

    When the number of models in one replica is larger than max_num_models_per_replica,
    the models will be unloaded using an LRU policy.

    If you want to release resources after the model is loaded, you can define
    a `__del__` method in your model class. The `__del__` method will be called when
    the model is unloaded.

    Example:

    .. code-block:: python

            from ray import serve

            @serve.deployment
            class MultiplexedDeployment:

                def __init__(self):
                    # Define s3 base path to load models.
                    self.s3_base_path = "s3://my_bucket/my_models"

                @serve.multiplexed(max_num_models_per_replica=5)
                async def load_model(self, model_id: str) -> Any:
                    # Load model with the given tag
                    # You can use any model loading library here
                    # and return the loaded model. load_from_s3 is
                    # a placeholder function.
                    return load_from_s3(model_id)

                async def __call__(self, request):
                    # Get the model_id from the request context.
                    model_id = serve.get_multiplexed_model_id()
                    # Load the model for the requested model_id.
                    # If the model is already cached locally,
                    # this will just be a dictionary lookup.
                    model = await self.load_model(model_id)
                    return model(request)


    Args:
        max_num_models_per_replica: the maximum number of models
            to be loaded on each replica. By default, it is 3, which
            means that each replica can cache up to 3 models. You can
            set it to a larger number if you have enough memory on
            the node resource, in opposite, you can set it to a smaller
            number if you want to save memory on the node resource.
    """
    if func is not None:
        if not callable(func):
            raise TypeError('The `multiplexed` decorator must be used with a function or method.')
        if not inspect.iscoroutinefunction(func):
            raise TypeError('@serve.multiplexed can only be used to decorate async functions or methods.')
        signature = inspect.signature(func)
        if len(signature.parameters) == 0 or len(signature.parameters) > 2:
            raise TypeError("@serve.multiplexed can only be used to decorate functions or methods with at least one 'model_id: str' argument.")
    if not isinstance(max_num_models_per_replica, int):
        raise TypeError('max_num_models_per_replica must be an integer.')
    if max_num_models_per_replica != -1 and max_num_models_per_replica <= 0:
        raise ValueError('max_num_models_per_replica must be positive.')

    def _multiplex_decorator(func: Callable):

        @wraps(func)
        async def _multiplex_wrapper(*args):
            args_check_error_msg = 'Functions decorated with `@serve.multiplexed` must take exactly onethe multiplexed model ID (str), but got {}'
            if not args:
                raise TypeError(args_check_error_msg.format('no arguments are provided.'))
            self = extract_self_if_method_call(args, func)
            if self is None:
                if len(args) != 1:
                    raise TypeError(args_check_error_msg.format('more than one arguments.'))
                multiplex_object = func
                model_id = args[0]
            else:
                if len(args) != 2:
                    raise TypeError(args_check_error_msg.format('more than one arguments.'))
                multiplex_object = self
                model_id = args[1]
            multiplex_attr = '__serve_multiplex_wrapper'
            if not hasattr(multiplex_object, multiplex_attr):
                model_multiplex_wrapper = _ModelMultiplexWrapper(func, self, max_num_models_per_replica)
                setattr(multiplex_object, multiplex_attr, model_multiplex_wrapper)
            else:
                model_multiplex_wrapper = getattr(multiplex_object, multiplex_attr)
            return await model_multiplex_wrapper.load_model(model_id)
        return _multiplex_wrapper
    return _multiplex_decorator(func) if callable(func) else _multiplex_decorator