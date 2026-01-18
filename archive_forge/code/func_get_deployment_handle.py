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
@DeveloperAPI
def get_deployment_handle(deployment_name: str, app_name: Optional[str]=None) -> DeploymentHandle:
    """Get a handle to a deployment by name.

    This is a developer API and is for advanced Ray users and library developers.

    Args:
        deployment_name: Name of deployment to get a handle to.
        app_name: Application in which deployment resides. If calling
            from inside a Serve application and `app_name` is not
            specified, this will default to the application from which
            this API is called.

    Raises:
        RayServeException: If no Serve controller is running, or if
            calling from outside a Serve application and no application
            name is specified.

    The following example gets the handle to the ingress deployment of
    an application, which is equivalent to using `serve.get_app_handle`.

    .. testcode::

            import ray
            from ray import serve

            @serve.deployment
            def f(val: int) -> int:
                return val * 2

            serve.run(f.bind(), name="my_app")
            handle = serve.get_deployment_handle("f", app_name="my_app")
            assert handle.remote(3).result() == 6

            serve.shutdown()

    The following example demonstrates how you can use this API to get
    the handle to a non-ingress deployment in an application.

    .. testcode::

            import ray
            from ray import serve
            from ray.serve.handle import DeploymentHandle

            @serve.deployment
            class Multiplier:
                def __init__(self, multiple: int):
                    self._multiple = multiple

                def __call__(self, val: int) -> int:
                    return val * self._multiple

            @serve.deployment
            class Adder:
                def __init__(self, handle: DeploymentHandle, increment: int):
                    self._handle = handle.options(use_new_handle_api=True)
                    self._increment = increment

                async def __call__(self, val: int) -> int:
                    return await self._handle.remote(val) + self._increment


            # The app calculates 2 * x + 3
            serve.run(Adder.bind(Multiplier.bind(2), 3), name="math_app")
            handle = serve.get_app_handle("math_app")
            assert handle.remote(5).result() == 13

            # Get handle to Multiplier only
            handle = serve.get_deployment_handle("Multiplier", app_name="math_app")
            assert handle.remote(5).result() == 10

            serve.shutdown()
    """
    client = _get_global_client()
    internal_replica_context = _get_internal_replica_context()
    if app_name is None:
        if internal_replica_context is None:
            raise RayServeException('Please specify an application name when getting a deployment handle outside of a Serve application.')
        else:
            app_name = internal_replica_context.app_name
    ServeUsageTag.SERVE_GET_DEPLOYMENT_HANDLE_API_USED.record('1')
    sync = internal_replica_context is None
    return client.get_handle(deployment_name, app_name, sync=sync, use_new_handle_api=True)