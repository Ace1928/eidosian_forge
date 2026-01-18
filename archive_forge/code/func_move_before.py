import copy
import inspect
import logging
from typing import Any, Callable, cast, Dict, List, Optional, Set, Tuple, Type
import torch.nn as nn
from torch import fx
from torch.distributed._spmd.graph_utils import (
from torch.distributed._spmd.partial_lower import partial_lower
from torch.fx.graph import _PyTreeCodeGen, PythonCode
from torch.fx.node import Argument
from torch.profiler import record_function
from torch.utils import _pytree as pytree
from torch.utils._pytree import tree_flatten, tree_map, tree_map_only, tree_unflatten
def move_before(self, nodes: List[fx.Node], target_node: fx.Node) -> None:
    for graph in self._all_graphs:
        actual_nodes = [self._lookup_node(node, graph) for node in nodes]
        actual_target_node = self._lookup_node(target_node, graph)
        assert actual_target_node is not None
        for actual_node in actual_nodes:
            actual_target_node.prepend(actual_node)