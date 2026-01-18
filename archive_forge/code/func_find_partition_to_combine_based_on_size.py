import operator
from collections import deque
from typing import Dict, List, Set, NamedTuple, Tuple, Deque
import torch
from torch.fx.passes.graph_manipulation import get_size_of_all_nodes
from torch.fx.experimental.partitioner_utils import (
from torch.fx.graph_module import GraphModule
from torch.fx.node import Node, map_arg
from torch.fx.passes.split_module import split_module
def find_partition_to_combine_based_on_size(sorted_partitions: List[Partition], available_mem_bytes: int, partitions: List[Partition]) -> Tuple[bool, List[Partition]]:
    """step 1 in combine_partition_based_on_size()"""
    find_combination = False
    smallest_partition = sorted_partitions.pop(0)
    for p in sorted_partitions[::-1]:
        if abs(smallest_partition.bfs_level - p.bfs_level) <= 1:
            mem_bytes_needed = calculate_mem_bytes_needed(p, smallest_partition)
            if mem_bytes_needed <= available_mem_bytes:
                combine_two_partitions(p, smallest_partition, self.partitions)
                partitions.remove(smallest_partition)
                partitions.remove(p)
                partitions.append(self.partitions[-1])
                find_combination = True
                break
    return (find_combination, partitions)