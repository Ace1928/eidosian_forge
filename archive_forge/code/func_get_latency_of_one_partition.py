from enum import Enum
from typing import NamedTuple, Dict, List, Set
from torch.fx.node import Node, map_arg
def get_latency_of_one_partition(partition: Partition, node_to_latency_mapping: Dict[Node, NodeLatency]) -> PartitionLatency:
    """Given a partition and its nodes' latency, return a PartitionLatency for this partition"""

    def get_top_nodes(partition: Partition) -> List[Node]:
        """Given a partition, return a list of nodes on the top bfs level"""
        top_nodes: List[Node] = []
        for node in partition.nodes:
            if node.op in {'placeholder', 'get_attr'}:
                continue
            input_nodes: Dict[Node, None] = {}
            map_arg(node.args, input_nodes.setdefault)
            map_arg(node.kwargs, input_nodes.setdefault)
            if not any((n in partition.nodes and n.op not in {'placeholder', 'get_attr'} for n in input_nodes)):
                top_nodes.append(node)
        return top_nodes

    def dfs_helper(node: Node, partition_latency) -> PartitionLatency:
        """Given a top node of a partition, this function returns
        the latency of the critical path in the partition
        """
        node_latency = node_to_latency_mapping[node]
        overall_latency_sec = partition_latency.overall_latency_sec + max(node_latency.computer_latency_sec, node_latency.mem_latency_sec)
        mem_latency_sec = partition_latency.mem_latency_sec + node_latency.mem_latency_sec
        computer_latency_sec = partition_latency.computer_latency_sec + node_latency.computer_latency_sec
        users = set(node.users).intersection(partition.nodes)
        if users:
            max_latency = PartitionLatency(mem_latency_sec=0.0, computer_latency_sec=0.0, overall_latency_sec=0.0)
            for n in users:
                new_partition_latency = dfs_helper(n, PartitionLatency(mem_latency_sec, computer_latency_sec, overall_latency_sec))
                if new_partition_latency.overall_latency_sec > max_latency.overall_latency_sec:
                    max_latency = new_partition_latency
            return max_latency
        return PartitionLatency(mem_latency_sec, computer_latency_sec, overall_latency_sec)
    top_nodes = get_top_nodes(partition)
    critical_path_latency = PartitionLatency(mem_latency_sec=0.0, computer_latency_sec=0.0, overall_latency_sec=0.0)
    for node in top_nodes:
        partition_latency = dfs_helper(node, PartitionLatency(mem_latency_sec=0.0, computer_latency_sec=0.0, overall_latency_sec=0.0))
        if partition_latency.overall_latency_sec > critical_path_latency.overall_latency_sec:
            critical_path_latency = partition_latency
    return critical_path_latency