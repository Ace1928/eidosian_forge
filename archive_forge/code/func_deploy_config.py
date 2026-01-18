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
def deploy_config(self, name: str, app_config: ServeApplicationSchema, deployment_time: float=0, target_capacity: Optional[float]=None, target_capacity_direction: Optional[TargetCapacityDirection]=None) -> None:
    """Deploy application from config."""
    if name not in self._application_states:
        self._application_states[name] = ApplicationState(name, self._deployment_state_manager, endpoint_state=self._endpoint_state, save_checkpoint_func=self._save_checkpoint_func)
    ServeUsageTag.NUM_APPS.record(str(len(self._application_states)))
    self._application_states[name].deploy_config(app_config, target_capacity, target_capacity_direction, deployment_time=deployment_time)