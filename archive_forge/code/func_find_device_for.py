import operator
from collections import deque
from typing import Dict, List, Set, NamedTuple, Tuple, Deque
import torch
from torch.fx.passes.graph_manipulation import get_size_of_all_nodes
from torch.fx.experimental.partitioner_utils import (
from torch.fx.graph_module import GraphModule
from torch.fx.node import Node, map_arg
from torch.fx.passes.split_module import split_module
def find_device_for(partition: Partition):
    """Given a partition, find a logical device for the partition
        The algorithm is to put the partition on the device
        that has just enough mem left for that partition.
        device_to_left_mem_bytes is a dictionary between device and its left mem size
        sorted by its left mem size
        """
    for d in device_to_left_mem_bytes:
        extra_size_needed = calculate_extra_mem_bytes_needed_for(partition, device_to_partitions[d])
        if extra_size_needed < device_to_left_mem_bytes[d]:
            device_to_partitions[d].append(partition)
            partition.logical_device_ids.append(d.logical_id)
            device_to_left_mem_bytes[d] -= extra_size_needed
            return True
    return False