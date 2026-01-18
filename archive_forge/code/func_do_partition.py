import operator
from collections import deque
from typing import Dict, List, Set, NamedTuple, Tuple, Deque
import torch
from torch.fx.passes.graph_manipulation import get_size_of_all_nodes
from torch.fx.experimental.partitioner_utils import (
from torch.fx.graph_module import GraphModule
from torch.fx.node import Node, map_arg
from torch.fx.passes.split_module import split_module
def do_partition(self) -> GraphModule:
    """Return a new fx module with submodule nodes (partitions)."""
    module_with_submodules = split_module(self.graph_module, self.torch_module, lambda node: self.node_to_partition[node])
    return module_with_submodules