import operator
from collections import deque
from typing import Dict, List, Set, NamedTuple, Tuple, Deque
import torch
from torch.fx.passes.graph_manipulation import get_size_of_all_nodes
from torch.fx.experimental.partitioner_utils import (
from torch.fx.graph_module import GraphModule
from torch.fx.node import Node, map_arg
from torch.fx.passes.split_module import split_module
def cost_aware_partition(self, transfer_rate_bytes_per_sec: float, node_to_latency_mapping: Dict[Node, NodeLatency]) -> None:
    """This method is to partition the fx module based on the cost.
        The cost is the total latency of running the whole fx module.
        In partitioner_utils.py, the cost model is built.
        The cost aware partition algorithm is:
        #1. At every beginning, each node is a partition.
            Then we map all the partitions to the devices
            and calculate the cost
        #2. Then try to pre-combine any two of the partitions if the two
            partitions can be combined.
            (the bfs level is less than 2 or two partitions are connected and
            can find partition to device mapping)
            See if any partition pair could reduce the current cost.
            Choose the pair that shows the minimum cost and then combine them
        #3. Repeat #2 until the cost cannot be reduced.
        """

    def try_combining_partitions(p0_index, p1_index, partitions) -> float:
        """Given two partitions and a list of partitions, combine these two partitions
            and see what is the cost of the modified partition list
            """
        p0 = partitions[p0_index]
        p1 = partitions[p1_index]
        "If two partitions' bfs level are less than 2 or two partitions are connected to each other,\n               then they can be combined\n            "
        if abs(p0.bfs_level - p1.bfs_level) <= 1 or p0 in p1.parents or p0 in p1.children:
            combine_two_partitions(p0, p1, partitions)
            if check_dependency(partitions[-1]):
                return float('inf')
            reset_partition_device(partitions)
            found_deivce = get_device_to_partitions_mapping(partitions, self.devices)
            if not found_deivce:
                return float('inf')
            partition_to_latency_mapping = get_partition_to_latency_mapping(partitions, node_to_latency_mapping)
            cost = get_latency_of_partitioned_graph(partitions, partition_to_latency_mapping, transfer_rate_bytes_per_sec)
            return cost
        return float('inf')

    def search_combination(transfer_rate_bytes_per_sec, node_to_latency_mapping) -> bool:
        """Given transfer rate between partitions and each node's latency,
            find two partitions to combine so the cost of the partitions can
            be reduced.
            The algorithm is :
            1. Go through all the partition pairs and see
            if any pair of partitions can be combined.
            2. Calculate the cost after the combination.
            3. Select the minimum cost and combine its corresponding partition pair.
            """
        partition_to_latency_mapping = get_partition_to_latency_mapping(self.partitions, node_to_latency_mapping)
        cost = get_latency_of_partitioned_graph(self.partitions, partition_to_latency_mapping, transfer_rate_bytes_per_sec)
        if len(self.partitions) == 1:
            return False
        partition_pair: List[int] = []
        for i in range(len(self.partitions) - 1):
            for j in range(i + 1, len(self.partitions)):
                new_cost = try_combining_partitions(i, j, self.partitions[:])
                if new_cost <= cost:
                    partition_pair = [i, j]
                    cost = new_cost
                reorganize_partitions(self.partitions)
        if len(partition_pair) != 0:
            p0 = self.partitions[partition_pair[0]]
            p1 = self.partitions[partition_pair[1]]
            combine_two_partitions(p0, p1, self.partitions)
        get_bfs_level_partition(self.partitions)
        reset_partition_device(self.partitions)
        get_device_to_partitions_mapping(self.partitions, self.devices)
        return len(partition_pair) != 0
    for node in self.graph_module.graph.nodes:
        if node.op not in {'placeholder', 'get_attr', 'output'}:
            self.create_single_node_partition(node)
    set_parents_and_children(self.partitions)
    get_bfs_level_partition(self.partitions)
    find_combination = True
    while find_combination:
        find_combination = search_combination(transfer_rate_bytes_per_sec, node_to_latency_mapping)
    reorganize_partitions(self.partitions)
    self.node_to_partition = get_node_to_partition_mapping(self.partitions)
    return