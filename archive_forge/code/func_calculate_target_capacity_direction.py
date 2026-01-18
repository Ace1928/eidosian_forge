import asyncio
import logging
import marshal
import os
import pickle
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
import ray
from ray._private.resource_spec import HEAD_NODE_RESOURCE_NAME
from ray._private.utils import run_background_task
from ray._raylet import GcsClient
from ray.actor import ActorHandle
from ray.serve._private.application_state import ApplicationStateManager
from ray.serve._private.common import (
from ray.serve._private.constants import (
from ray.serve._private.default_impl import create_cluster_node_info_cache
from ray.serve._private.deploy_utils import deploy_args_to_deployment_info
from ray.serve._private.deployment_info import DeploymentInfo
from ray.serve._private.deployment_state import DeploymentStateManager
from ray.serve._private.endpoint_state import EndpointState
from ray.serve._private.logging_utils import (
from ray.serve._private.long_poll import LongPollHost, LongPollNamespace
from ray.serve._private.proxy_state import ProxyStateManager
from ray.serve._private.storage.kv_store import RayInternalKVStore
from ray.serve._private.usage import ServeUsageTag
from ray.serve._private.utils import (
from ray.serve.config import HTTPOptions, gRPCOptions
from ray.serve.generated.serve_pb2 import (
from ray.serve.generated.serve_pb2 import EndpointInfo as EndpointInfoProto
from ray.serve.generated.serve_pb2 import EndpointSet
from ray.serve.schema import (
from ray.util import metrics
def calculate_target_capacity_direction(curr_config: Optional[ServeDeploySchema], new_config: ServeDeploySchema, curr_target_capacity_direction: Optional[float]) -> Optional[TargetCapacityDirection]:
    """Compares two Serve configs to calculate the next scaling direction."""
    curr_target_capacity = None
    next_target_capacity_direction = None
    if curr_config is not None and applications_match(curr_config, new_config):
        curr_target_capacity = curr_config.target_capacity
        next_target_capacity = new_config.target_capacity
        if curr_target_capacity == next_target_capacity:
            next_target_capacity_direction = curr_target_capacity_direction
        elif curr_target_capacity is None and next_target_capacity is not None:
            next_target_capacity_direction = TargetCapacityDirection.DOWN
        elif next_target_capacity is None:
            next_target_capacity_direction = None
        elif curr_target_capacity < next_target_capacity:
            next_target_capacity_direction = TargetCapacityDirection.UP
        else:
            next_target_capacity_direction = TargetCapacityDirection.DOWN
    elif new_config.target_capacity is not None:
        next_target_capacity_direction = TargetCapacityDirection.UP
    else:
        next_target_capacity_direction = None
    return next_target_capacity_direction