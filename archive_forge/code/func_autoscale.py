import json
import logging
import math
import os
import random
import time
import traceback
from collections import defaultdict
from copy import copy
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import ray
from ray import ObjectRef, cloudpickle
from ray.actor import ActorHandle
from ray.exceptions import RayActorError, RayError, RayTaskError, RuntimeEnvSetupError
from ray.serve import metrics
from ray.serve._private import default_impl
from ray.serve._private.autoscaling_metrics import InMemoryMetricsStore
from ray.serve._private.cluster_node_info_cache import ClusterNodeInfoCache
from ray.serve._private.common import (
from ray.serve._private.config import DeploymentConfig
from ray.serve._private.constants import (
from ray.serve._private.deployment_info import DeploymentInfo
from ray.serve._private.deployment_scheduler import (
from ray.serve._private.long_poll import LongPollHost, LongPollNamespace
from ray.serve._private.storage.kv_store import KVStoreBase
from ray.serve._private.usage import ServeUsageTag
from ray.serve._private.utils import (
from ray.serve._private.version import DeploymentVersion, VersionedReplica
from ray.serve.generated.serve_pb2 import DeploymentLanguage
from ray.serve.schema import (
from ray.util.placement_group import PlacementGroup
def autoscale(self, current_handle_queued_queries: int) -> int:
    """Autoscale the deployment based on metrics.

        Args:
            current_handle_queued_queries: The number of handle queued queries,
                if there are multiple handles, the max number of queries at
                a single handle should be passed in
        """
    if self._target_state.deleting:
        return
    current_num_ongoing_requests = self.get_replica_current_ongoing_requests()
    autoscaling_policy = self._target_state.info.autoscaling_policy
    decision_num_replicas = autoscaling_policy.get_decision_num_replicas(curr_target_num_replicas=self._target_state.target_num_replicas, current_num_ongoing_requests=current_num_ongoing_requests, current_handle_queued_queries=current_handle_queued_queries, target_capacity=self._target_state.info.target_capacity, target_capacity_direction=self._target_state.info.target_capacity_direction)
    if decision_num_replicas == self._target_state.target_num_replicas:
        return
    logger.info(f'Autoscaling replicas for deployment {self.deployment_name} in application {self.app_name} to {decision_num_replicas}. current_num_ongoing_requests: {current_num_ongoing_requests}, current handle queued queries: {current_handle_queued_queries}.')
    new_info = copy(self._target_state.info)
    new_info.version = self._target_state.version.code_version
    allow_scaling_statuses = self._is_within_autoscaling_bounds() or self.curr_status_info.status_trigger != DeploymentStatusTrigger.CONFIG_UPDATE_STARTED
    self._set_target_state(new_info, decision_num_replicas, status_trigger=DeploymentStatusTrigger.AUTOSCALING, allow_scaling_statuses=allow_scaling_statuses)