import logging
import random
import time
from functools import wraps
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
import ray
from ray.actor import ActorHandle
from ray.serve._private.common import (
from ray.serve._private.config import DeploymentConfig, ReplicaConfig
from ray.serve._private.constants import (
from ray.serve._private.controller import ServeController
from ray.serve._private.deploy_utils import get_deploy_args
from ray.serve._private.deployment_info import DeploymentInfo
from ray.serve.config import HTTPOptions
from ray.serve.exceptions import RayServeException
from ray.serve.generated.serve_pb2 import (
from ray.serve.generated.serve_pb2 import (
from ray.serve.generated.serve_pb2 import StatusOverview as StatusOverviewProto
from ray.serve.handle import DeploymentHandle, RayServeHandle, RayServeSyncHandle
from ray.serve.schema import LoggingConfig, ServeApplicationSchema, ServeDeploySchema
@_ensure_connected
def log_deployment_update_status(self, name: str, version: str, updating: bool) -> str:
    tag = f'component=serve deployment={name}'
    if updating:
        msg = f"Updating deployment '{name}'"
        if version is not None:
            msg += f" to version '{version}'"
        logger.info(f'{msg}. {tag}')
    else:
        logger.info(f"Deployment '{name}' is already at version '{version}', not updating. {tag}")
    return tag