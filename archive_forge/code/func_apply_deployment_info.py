import logging
import time
import traceback
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple
import ray
from ray import cloudpickle
from ray._private.utils import import_attr
from ray.exceptions import RuntimeEnvSetupError
from ray.serve._private.common import (
from ray.serve._private.config import DeploymentConfig
from ray.serve._private.constants import SERVE_LOGGER_NAME
from ray.serve._private.deploy_utils import (
from ray.serve._private.deployment_info import DeploymentInfo
from ray.serve._private.deployment_state import DeploymentStateManager
from ray.serve._private.endpoint_state import EndpointState
from ray.serve._private.storage.kv_store import KVStoreBase
from ray.serve._private.usage import ServeUsageTag
from ray.serve._private.utils import (
from ray.serve.exceptions import RayServeException
from ray.serve.generated.serve_pb2 import DeploymentLanguage
from ray.serve.schema import DeploymentDetails, ServeApplicationSchema
from ray.types import ObjectRef
def apply_deployment_info(self, deployment_name: str, deployment_info: DeploymentInfo) -> None:
    """Deploys a deployment in the application."""
    route_prefix = deployment_info.route_prefix
    if route_prefix is not None and (not route_prefix.startswith('/')):
        raise RayServeException(f'Invalid route prefix "{route_prefix}", it must start with "/"')
    deployment_id = DeploymentID(deployment_name, self._name)
    self._deployment_state_manager.deploy(deployment_id, deployment_info)
    if deployment_info.route_prefix is not None:
        config = deployment_info.deployment_config
        self._endpoint_state.update_endpoint(deployment_id, EndpointInfo(route=deployment_info.route_prefix, app_is_cross_language=config.deployment_language != DeploymentLanguage.PYTHON))
    else:
        self._endpoint_state.delete_endpoint(deployment_id)