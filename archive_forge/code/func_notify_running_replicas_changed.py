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
def notify_running_replicas_changed(self) -> None:
    running_replica_infos = self.get_running_replica_infos()
    if set(self._last_notified_running_replica_infos) == set(running_replica_infos) and (not self._multiplexed_model_ids_updated):
        return
    self._long_poll_host.notify_changed((LongPollNamespace.RUNNING_REPLICAS, self._id), running_replica_infos)
    self._long_poll_host.notify_changed((LongPollNamespace.RUNNING_REPLICAS, self._id.name), running_replica_infos)
    self._last_notified_running_replica_infos = running_replica_infos
    self._multiplexed_model_ids_updated = False