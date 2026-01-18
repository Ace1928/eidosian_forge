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
def reserve_and_allocate_spread(self, strict_spreads: List[List[ResourceDict]], node_resources: List[ResourceDict], node_type_counts: Dict[NodeType, int], utilization_scorer: Callable[[NodeResources, ResourceDemands], Optional[UtilizationScore]]):
    """For each strict spread, attempt to reserve as much space as possible
        on the node, then allocate new nodes for the unfulfilled portion.

        Args:
            strict_spreads (List[List[ResourceDict]]): A list of placement
                groups which must be spread out.
            node_resources (List[ResourceDict]): Available node resources in
                the cluster.
            node_type_counts (Dict[NodeType, int]): The amount of each type of
                node pending or in the cluster.
            utilization_scorer: A function that, given a node
                type, its resources, and resource demands, returns what its
                utilization would be.

        Returns:
            Dict[NodeType, int]: Nodes to add.
            List[ResourceDict]: The updated node_resources after the method.
            Dict[NodeType, int]: The updated node_type_counts.

        """
    to_add = collections.defaultdict(int)
    for bundles in strict_spreads:
        unfulfilled, node_resources = get_bin_pack_residual(node_resources, bundles, strict_spread=True)
        max_to_add = self.max_workers + 1 - sum(node_type_counts.values())
        to_launch, _ = get_nodes_for(self.node_types, node_type_counts, self.head_node_type, max_to_add, unfulfilled, utilization_scorer=utilization_scorer, strict_spread=True)
        _inplace_add(node_type_counts, to_launch)
        _inplace_add(to_add, to_launch)
        new_node_resources = _node_type_counts_to_node_resources(self.node_types, to_launch)
        unfulfilled, including_reserved = get_bin_pack_residual(new_node_resources, unfulfilled, strict_spread=True)
        assert not unfulfilled
        node_resources += including_reserved
    return (to_add, node_resources, node_type_counts)