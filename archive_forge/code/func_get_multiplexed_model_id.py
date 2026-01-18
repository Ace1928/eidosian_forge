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
def get_multiplexed_model_id() -> str:
    """Get the multiplexed model ID for the current request.

    This is used with a function decorated with `@serve.multiplexed`
    to retrieve the model ID for the current request.

    .. code-block:: python

            import ray
            from ray import serve
            import requests

            # Set the multiplexed model id with the key
            # "ray_serve_multiplexed_model_id" in the request
            # headers when sending requests to the http proxy.
            requests.get("http://localhost:8000",
                headers={"ray_serve_multiplexed_model_id": "model_1"})
            # This can also be set when using `RayServeHandle`.
            handle.options(multiplexed_model_id="model_1").remote("blablabla")

            # In your deployment code, you can retrieve the model id from
            # `get_multiplexed_model_id()`.
            @serve.deployment
            def my_deployment_function(request):
                assert serve.get_multiplexed_model_id() == "model_1"
    """
    _request_context = ray.serve.context._serve_request_context.get()
    return _request_context.multiplexed_model_id