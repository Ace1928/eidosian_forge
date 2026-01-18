import operator
from collections import deque
from typing import Dict, List, Set, NamedTuple, Tuple, Deque
import torch
from torch.fx.passes.graph_manipulation import get_size_of_all_nodes
from torch.fx.experimental.partitioner_utils import (
from torch.fx.graph_module import GraphModule
from torch.fx.node import Node, map_arg
from torch.fx.passes.split_module import split_module
def combine_partitions_based_on_size(partitions: List[Partition], available_mem_bytes: int) -> None:
    """Combining small partitions together to keep as less partitions as possible.
            Here is an example of the algorithm to do this:
            Assume some partitions, we first sort them based on partition used memory size.
            [(partition_4, 1), (partition_3, 1), (partition_2, 2), (partition_1, 7), (partition_0, 9)]
            The available memory is 10.
            step 1: self.find_partition_to_combine_based_on_size()
            First, mark bfs level for each partition
            Second, look the smallest partition, partition_4: 10 - 1 = 9
            It means any partition has a used memory equal or less than 9 could combine this partition
            We go from the largest and selection partition_0.
            Check the bfs level for two partitions, if the level difference is less than 2,
            it can be combined.
            step 2: repeat step 1 until no partitions can be combined
            """
    find_combination = True
    while find_combination:
        sorted_partitions = sorted(partitions, key=lambda p: p.used_mem_bytes)
        get_bfs_level_partition(self.partitions)
        find_combination, partitions = find_partition_to_combine_based_on_size(sorted_partitions, available_mem_bytes, partitions)
    return