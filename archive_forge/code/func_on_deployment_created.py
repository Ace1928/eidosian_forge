import copy
import sys
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Set, Tuple
import ray
from ray.serve._private.cluster_node_info_cache import ClusterNodeInfoCache
from ray.serve._private.common import DeploymentID
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
def on_deployment_created(self, deployment_id: DeploymentID, scheduling_policy: SpreadDeploymentSchedulingPolicy) -> None:
    """Called whenever a new deployment is created."""
    assert deployment_id not in self._pending_replicas
    assert deployment_id not in self._launching_replicas
    assert deployment_id not in self._recovering_replicas
    assert deployment_id not in self._running_replicas
    self._deployments[deployment_id] = scheduling_policy