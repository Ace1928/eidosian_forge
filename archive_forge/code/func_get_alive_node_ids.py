from abc import ABC, abstractmethod
from typing import List, Optional, Set, Tuple
import ray
from ray._raylet import GcsClient
from ray.serve._private.constants import RAY_GCS_RPC_TIMEOUT_S
def get_alive_node_ids(self) -> Set[str]:
    """Get IDs of all live nodes in the cluster."""
    return {node_id for node_id, _ in self.get_alive_nodes()}