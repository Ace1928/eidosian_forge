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
def delete_apps(self, names: List[str], blocking: bool=True):
    if not names:
        return
    logger.info(f'Deleting app {names}')
    self._controller.delete_apps.remote(names)
    if blocking:
        start = time.time()
        while time.time() - start < 60:
            curr_statuses_bytes = ray.get(self._controller.get_serve_statuses.remote(names))
            all_deleted = True
            for cur_status_bytes in curr_statuses_bytes:
                cur_status = StatusOverview.from_proto(StatusOverviewProto.FromString(cur_status_bytes))
                if cur_status.app_status.status != ApplicationStatus.NOT_STARTED:
                    all_deleted = False
            if all_deleted:
                return
            time.sleep(CLIENT_POLLING_INTERVAL_S)
        else:
            raise TimeoutError(f"Some of these applications weren't deleted after 60s: {names}")