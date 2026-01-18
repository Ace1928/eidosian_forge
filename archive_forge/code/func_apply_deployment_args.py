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
def apply_deployment_args(self, name: str, deployment_args: List[Dict]) -> None:
    """Apply list of deployment arguments to application target state.

        This function should only be called if the app is being deployed
        through serve.run instead of from a config.

        Args:
            name: application name
            deployment_args_list: arguments for deploying a list of deployments.

        Raises:
            RayServeException: If the list of deployments is trying to
                use a route prefix that is already used by another application
        """
    live_route_prefixes: Dict[str, str] = {self._application_states[app_name].route_prefix: app_name for app_name, app_state in self._application_states.items() if app_state.route_prefix is not None and (not app_state.status == ApplicationStatus.DELETING) and (name != app_name)}
    for deploy_param in deployment_args:
        deploy_app_prefix = deploy_param.get('route_prefix')
        if deploy_app_prefix in live_route_prefixes:
            raise RayServeException(f'Prefix {deploy_app_prefix} is being used by application "{live_route_prefixes[deploy_app_prefix]}". Failed to deploy application "{name}".')
    if name not in self._application_states:
        self._application_states[name] = ApplicationState(name, self._deployment_state_manager, self._endpoint_state, self._save_checkpoint_func)
    ServeUsageTag.NUM_APPS.record(str(len(self._application_states)))
    deployment_infos = {params['deployment_name']: deploy_args_to_deployment_info(**params, app_name=name) for params in deployment_args}
    self._application_states[name].deploy(deployment_infos)