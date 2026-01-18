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
def graceful_stop(self) -> Duration:
    """Request the actor to exit gracefully.

        Returns the timeout after which to kill the actor.
        """
    try:
        handle = ray.get_actor(self._actor_name, namespace=SERVE_NAMESPACE)
        self._graceful_shutdown_ref = handle.prepare_for_shutdown.remote()
    except ValueError:
        pass
    return self.graceful_shutdown_timeout_s