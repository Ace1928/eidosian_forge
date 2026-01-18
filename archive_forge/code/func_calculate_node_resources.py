import collections
import copy
import logging
import os
from abc import abstractmethod
from functools import partial
from typing import Callable, Dict, List, Optional, Tuple
import ray
from ray._private.gcs_utils import PlacementGroupTableData
from ray.autoscaler._private.constants import (
from ray.autoscaler._private.loader import load_function_or_class
from ray.autoscaler._private.node_provider_availability_tracker import (
from ray.autoscaler._private.util import (
from ray.autoscaler.node_provider import NodeProvider
from ray.autoscaler.tags import (
from ray.core.generated.common_pb2 import PlacementStrategy
def calculate_node_resources(self, nodes: List[NodeID], pending_nodes: Dict[NodeID, int], unused_resources_by_ip: Dict[str, ResourceDict]) -> (List[ResourceDict], Dict[NodeType, int]):
    """Returns node resource list and node type counts.

        Counts the running nodes, pending nodes.
        Args:
             nodes: Existing nodes.
             pending_nodes: Pending nodes.
        Returns:
             node_resources: a list of running + pending resources.
                 E.g., [{"CPU": 4}, {"GPU": 2}].
             node_type_counts: running + pending workers per node type.
        """
    node_resources = []
    node_type_counts = collections.defaultdict(int)

    def add_node(node_type, available_resources=None):
        if node_type not in self.node_types:
            logger.error(f'''Missing entry for node_type {node_type} in cluster config: {self.node_types} under entry available_node_types. This node's resources will be ignored. If you are using an unmanaged node, manually set the {TAG_RAY_NODE_KIND} tag to "{NODE_KIND_UNMANAGED}" in your cloud provider's management console.''')
            return None
        available = copy.deepcopy(self.node_types[node_type]['resources'])
        if available_resources is not None:
            available = copy.deepcopy(available_resources)
        node_resources.append(available)
        node_type_counts[node_type] += 1
    for node_id in nodes:
        tags = self.provider.node_tags(node_id)
        if TAG_RAY_USER_NODE_TYPE in tags:
            node_type = tags[TAG_RAY_USER_NODE_TYPE]
            ip = self.provider.internal_ip(node_id)
            available_resources = unused_resources_by_ip.get(ip)
            add_node(node_type, available_resources)
    for node_type, count in pending_nodes.items():
        for _ in range(count):
            add_node(node_type)
    return (node_resources, node_type_counts)