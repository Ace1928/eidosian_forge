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
def get_nodes_to_launch(self, nodes: List[NodeID], launching_nodes: Dict[NodeType, int], resource_demands: List[ResourceDict], unused_resources_by_ip: Dict[NodeIP, ResourceDict], pending_placement_groups: List[PlacementGroupTableData], max_resources_by_ip: Dict[NodeIP, ResourceDict], ensure_min_cluster_size: List[ResourceDict], node_availability_summary: NodeAvailabilitySummary) -> (Dict[NodeType, int], List[ResourceDict]):
    """Given resource demands, return node types to add to the cluster.

        This method:
            (1) calculates the resources present in the cluster.
            (2) calculates the remaining nodes to add to respect min_workers
                constraint per node type.
            (3) for each strict spread placement group, reserve space on
                available nodes and launch new nodes if necessary.
            (4) calculates the unfulfilled resource bundles.
            (5) calculates which nodes need to be launched to fulfill all
                the bundle requests, subject to max_worker constraints.

        Args:
            nodes: List of existing nodes in the cluster.
            launching_nodes: Summary of node types currently being launched.
            resource_demands: Vector of resource demands from the scheduler.
            unused_resources_by_ip: Mapping from ip to available resources.
            pending_placement_groups: Placement group demands.
            max_resources_by_ip: Mapping from ip to static node resources.
            ensure_min_cluster_size: Try to ensure the cluster can fit at least
                this set of resources. This differs from resources_demands in
                that we don't take into account existing usage.

            node_availability_summary: A snapshot of the current
                NodeAvailabilitySummary.

        Returns:
            Dict of count to add for each node type, and residual of resources
            that still cannot be fulfilled.
        """
    utilization_scorer = partial(self.utilization_scorer, node_availability_summary=node_availability_summary)
    self._update_node_resources_from_runtime(nodes, max_resources_by_ip)
    node_resources: List[ResourceDict]
    node_type_counts: Dict[NodeType, int]
    node_resources, node_type_counts = self.calculate_node_resources(nodes, launching_nodes, unused_resources_by_ip)
    logger.debug('Cluster resources: {}'.format(node_resources))
    logger.debug('Node counts: {}'.format(node_type_counts))
    node_resources, node_type_counts, adjusted_min_workers = _add_min_workers_nodes(node_resources, node_type_counts, self.node_types, self.max_workers, self.head_node_type, ensure_min_cluster_size, utilization_scorer=utilization_scorer)
    logger.debug(f'Placement group demands: {pending_placement_groups}')
    placement_group_demand_vector, strict_spreads = placement_groups_to_resource_demands(pending_placement_groups)
    resource_demands = placement_group_demand_vector + resource_demands
    spread_pg_nodes_to_add, node_resources, node_type_counts = self.reserve_and_allocate_spread(strict_spreads, node_resources, node_type_counts, utilization_scorer)
    unfulfilled_placement_groups_demands, _ = get_bin_pack_residual(node_resources, placement_group_demand_vector)
    max_to_add = self.max_workers + 1 - sum(node_type_counts.values())
    pg_demands_nodes_max_launch_limit, _ = get_nodes_for(self.node_types, node_type_counts, self.head_node_type, max_to_add, unfulfilled_placement_groups_demands, utilization_scorer=utilization_scorer)
    placement_groups_nodes_max_limit = {node_type: spread_pg_nodes_to_add.get(node_type, 0) + pg_demands_nodes_max_launch_limit.get(node_type, 0) for node_type in self.node_types}
    unfulfilled, _ = get_bin_pack_residual(node_resources, resource_demands)
    logger.debug('Resource demands: {}'.format(resource_demands))
    logger.debug('Unfulfilled demands: {}'.format(unfulfilled))
    nodes_to_add_based_on_demand, final_unfulfilled = get_nodes_for(self.node_types, node_type_counts, self.head_node_type, max_to_add, unfulfilled, utilization_scorer=utilization_scorer)
    logger.debug('Final unfulfilled: {}'.format(final_unfulfilled))
    total_nodes_to_add = {}
    for node_type in self.node_types:
        nodes_to_add = adjusted_min_workers.get(node_type, 0) + spread_pg_nodes_to_add.get(node_type, 0) + nodes_to_add_based_on_demand.get(node_type, 0)
        if nodes_to_add > 0:
            total_nodes_to_add[node_type] = nodes_to_add
    total_nodes_to_add = self._get_concurrent_resource_demand_to_launch(total_nodes_to_add, unused_resources_by_ip.keys(), nodes, launching_nodes, adjusted_min_workers, placement_groups_nodes_max_limit)
    logger.debug('Node requests: {}'.format(total_nodes_to_add))
    return (total_nodes_to_add, final_unfulfilled)