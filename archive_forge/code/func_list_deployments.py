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
def list_deployments() -> Dict[str, Deployment]:
    """Returns a dictionary of all active 1.x deployments.

    Dictionary maps deployment name to Deployment objects.
    """
    infos = _get_global_client().list_deployments_v1()
    deployments = {}
    for name, (deployment_info, route_prefix) in infos.items():
        deployments[name] = Deployment(name, deployment_info.deployment_config, deployment_info.replica_config, version=deployment_info.version, route_prefix=route_prefix, _internal=True)
    return deployments