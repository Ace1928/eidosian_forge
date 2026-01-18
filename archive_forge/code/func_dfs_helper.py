from enum import Enum
from typing import NamedTuple, Dict, List, Set
from torch.fx.node import Node, map_arg
def dfs_helper(partition: Partition, latency_so_far_sec: float) -> float:
    """This function helps to recursively get the latency of a path of partitions"""
    latency_so_far_sec += partition_to_latency_mapping[partition].overall_latency_sec
    children = partition.children
    if partition.children:
        max_latency_sec = 0.0
        for child in partition.children:
            comm_latency_sec = get_comm_latency_between(partition, child, transfer_rate_bytes_per_sec)
            new_latency_sec = dfs_helper(child, latency_so_far_sec + comm_latency_sec)
            if new_latency_sec > max_latency_sec:
                max_latency_sec = new_latency_sec
        return max_latency_sec
    return latency_so_far_sec