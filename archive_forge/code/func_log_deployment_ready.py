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
def log_deployment_ready(self, name: str, version: str, url: str, tag: str) -> None:
    if url is not None:
        url_part = f' at `{url}`'
    else:
        url_part = ''
    logger.info(f"Deployment '{name}{(':' + version if version else '')}' is ready{url_part}. {tag}")