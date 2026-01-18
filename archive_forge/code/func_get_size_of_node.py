from typing import Any, Dict, List, NamedTuple, Optional
import torch
from torch.fx._compatibility import compatibility
from torch.fx.graph import Graph
from torch.fx.graph_module import GraphModule
from torch.fx.node import (
from torch.fx.passes.shape_prop import ShapeProp
@compatibility(is_backward_compatible=False)
def get_size_of_node(fx_module: GraphModule, node: Node) -> size_bytes:
    """Given a node with node.dtype and node.shape, return its total size and its output size.
    total_size = weights + bias + output_size
    """
    total_num_of_elems = 0
    if node.op == 'call_module':
        submodule_dict = dict(fx_module.named_modules())
        submodule = submodule_dict[node.target]
        parameters = submodule.named_parameters()
        for name, p in parameters:
            total_num_of_elems += p.numel()
    tensor_meta = get_tensor_meta(node)
    output_elem = tensor_meta.shape.numel()
    total_num_of_elems += output_elem
    if tensor_meta.is_quantized:
        size_per_elem_bytes = torch._empty_affine_quantized([], dtype=tensor_meta.dtype).element_size()
    else:
        size_per_elem_bytes = torch.tensor([], dtype=tensor_meta.dtype).element_size()
    total_size = size_per_elem_bytes * total_num_of_elems
    output_size = size_per_elem_bytes * output_elem
    return size_bytes(output_size, total_size)