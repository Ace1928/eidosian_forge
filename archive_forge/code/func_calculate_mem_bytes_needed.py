import operator
from collections import deque
from typing import Dict, List, Set, NamedTuple, Tuple, Deque
import torch
from torch.fx.passes.graph_manipulation import get_size_of_all_nodes
from torch.fx.experimental.partitioner_utils import (
from torch.fx.graph_module import GraphModule
from torch.fx.node import Node, map_arg
from torch.fx.passes.split_module import split_module
def calculate_mem_bytes_needed(p1, p2):
    """Given two partitions, calculate how many mem bytes
            are needed if two partitions are combined
            """
    nodes = p1.nodes.union(p2.nodes)
    mem_bytes_needed = 0
    for node in nodes:
        mem_bytes_needed += get_extra_size_of(node, nodes)
    return mem_bytes_needed