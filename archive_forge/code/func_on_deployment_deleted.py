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
def on_deployment_deleted(self, deployment_id: DeploymentID) -> None:
    """Called whenever a deployment is deleted."""
    assert not self._pending_replicas[deployment_id]
    self._pending_replicas.pop(deployment_id, None)
    assert not self._launching_replicas[deployment_id]
    self._launching_replicas.pop(deployment_id, None)
    assert not self._recovering_replicas[deployment_id]
    self._recovering_replicas.pop(deployment_id, None)
    assert not self._running_replicas[deployment_id]
    self._running_replicas.pop(deployment_id, None)
    del self._deployments[deployment_id]