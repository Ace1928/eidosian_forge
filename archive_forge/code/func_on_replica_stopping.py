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
def on_replica_stopping(self, deployment_id: DeploymentID, replica_name: str) -> None:
    """Called whenever a deployment replica is being stopped."""
    self._pending_replicas[deployment_id].pop(replica_name, None)
    self._launching_replicas[deployment_id].pop(replica_name, None)
    self._recovering_replicas[deployment_id].discard(replica_name)
    self._running_replicas[deployment_id].pop(replica_name, None)