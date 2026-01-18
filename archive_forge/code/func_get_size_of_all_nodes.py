from typing import Any, Dict, List, NamedTuple, Optional
import torch
from torch.fx._compatibility import compatibility
from torch.fx.graph import Graph
from torch.fx.graph_module import GraphModule
from torch.fx.node import (
from torch.fx.passes.shape_prop import ShapeProp
@compatibility(is_backward_compatible=False)
def get_size_of_all_nodes(fx_module: GraphModule, args: Optional[List[torch.Tensor]]=None) -> None:
    """Given a fx graph module, update each node with its total size (weights + bias + output)
    and its output_size(output). For a non-module node, the total size is the output size.
    return total size"""
    if args is not None:
        ShapeProp(fx_module).propagate(*args)
    total_size_of_graph = 0.0
    for node in fx_module.graph.nodes:
        if node.op == 'output':
            break
        node.size_bytes = get_size_of_node(fx_module, node)
    return