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
def check_ready(self) -> Tuple[ReplicaStartupStatus, Optional[str]]:
    """
        Check if current replica has started by making ray API calls on
        relevant actor / object ref.

        Replica initialization calls __init__(), reconfigure(), and check_health().

        Returns:
            state (ReplicaStartupStatus):
                PENDING_ALLOCATION: replica is waiting for a worker to start
                PENDING_INITIALIZATION: replica initialization hasn't finished.
                FAILED: replica initialization failed.
                SUCCEEDED: replica initialization succeeded.
            error_msg:
                None: for PENDING_ALLOCATION, PENDING_INITIALIZATION or SUCCEEDED states
                str: for FAILED state
        """
    if self._allocated_obj_ref is None or not check_obj_ref_ready_nowait(self._allocated_obj_ref):
        return (ReplicaStartupStatus.PENDING_ALLOCATION, None)
    if not self._is_cross_language:
        try:
            self._pid, self._actor_id, self._worker_id, self._node_id, self._node_ip, self._log_file_path = ray.get(self._allocated_obj_ref)
        except RayTaskError as e:
            logger.exception(f"Exception in replica '{self._replica_tag}', the replica will be stopped.")
            return (ReplicaStartupStatus.FAILED, str(e.as_instanceof_cause()))
        except RuntimeEnvSetupError as e:
            msg = f"Exception when allocating replica '{self._replica_tag}': {str(e)}"
            logger.exception(msg)
            return (ReplicaStartupStatus.FAILED, msg)
        except Exception:
            msg = f"Exception when allocating replica '{self._replica_tag}':\n" + traceback.format_exc()
            logger.exception(msg)
            return (ReplicaStartupStatus.FAILED, msg)
    replica_ready = check_obj_ref_ready_nowait(self._ready_obj_ref)
    if not replica_ready:
        return (ReplicaStartupStatus.PENDING_INITIALIZATION, None)
    else:
        try:
            if self._is_cross_language:
                return (ReplicaStartupStatus.SUCCEEDED, None)
            if not self._deployment_is_cross_language:
                _, self._version = ray.get(self._ready_obj_ref)
        except RayTaskError as e:
            logger.exception(f"Exception in replica '{self._replica_tag}', the replica will be stopped.")
            return (ReplicaStartupStatus.FAILED, str(e.as_instanceof_cause()))
        except Exception as e:
            logger.exception(f"Exception in replica '{self._replica_tag}', the replica will be stopped.")
            return (ReplicaStartupStatus.FAILED, repr(e))
    return (ReplicaStartupStatus.SUCCEEDED, None)