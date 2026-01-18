from abc import ABC, abstractmethod
from typing import List, Optional, Set, Tuple
import ray
from ray._raylet import GcsClient
from ray.serve._private.constants import RAY_GCS_RPC_TIMEOUT_S
def get_active_node_ids(self) -> Set[str]:
    """Get IDs of all active nodes in the cluster.

        A node is active if it's schedulable for new tasks and actors.
        """
    return self.get_alive_node_ids() - self.get_draining_node_ids()