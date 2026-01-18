import operator
from collections import deque
from typing import Dict, List, Set, NamedTuple, Tuple, Deque
import torch
from torch.fx.passes.graph_manipulation import get_size_of_all_nodes
from torch.fx.experimental.partitioner_utils import (
from torch.fx.graph_module import GraphModule
from torch.fx.node import Node, map_arg
from torch.fx.passes.split_module import split_module
def set_parents_and_children(partitions: List[Partition]) -> None:
    """Given a list of partitions, mark parents and children for each partition"""
    for partition in partitions:
        partition.children = set()
        partition.parents = set()
    for partition in partitions:
        for node in partition.nodes:
            users = node.users
            for n in users:
                for p in partitions:
                    if p != partition and n in p.nodes and (node not in p.nodes):
                        partition.children.add(p)
                        p.parents.add(partition)
    return