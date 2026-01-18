import operator
from collections import deque
from typing import Dict, List, Set, NamedTuple, Tuple, Deque
import torch
from torch.fx.passes.graph_manipulation import get_size_of_all_nodes
from torch.fx.experimental.partitioner_utils import (
from torch.fx.graph_module import GraphModule
from torch.fx.node import Node, map_arg
from torch.fx.passes.split_module import split_module
def swap_nodes(n0, n1, p0, p1):
    if n0 is not None:
        p0.remove_node(n0)
        p1.add_node(n0)
    if n1 is not None:
        p0.add_node(n1)
        p1.remove_node(n1)