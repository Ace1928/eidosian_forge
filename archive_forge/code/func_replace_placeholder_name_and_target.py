from __future__ import annotations
import collections
import re
from typing import Callable, Dict, Optional, Tuple
import torch.fx
import torch.fx.traceback as fx_traceback
from torch.onnx._internal import _beartype
@_beartype.beartype
def replace_placeholder_name_and_target(module: torch.fx.GraphModule, reference_module: torch.fx.GraphModule):
    """Replace the argument names in module with those in reference_module.

    This function assumes the two modules have the same signature structure.
    The caller is responsible for ensuring this. Otherwise, the behavior of this
    function is undefined. This function only does minimal sanity check that the two
    modules have the same number of arguments.

    Name conflicts between new names and existing node names in the graph are handled.
    Check the documentation of :func:`set_node_name` for more details.

    Raises:
        RuntimeError: If the two modules have different number of arguments.
    """
    placeholders = [node for node in module.graph.nodes if node.op == 'placeholder']
    reference_placeholders = [node for node in reference_module.graph.nodes if node.op == 'placeholder']
    if len(placeholders) != len(reference_placeholders):
        raise RuntimeError(f'The two modules have different number of arguments. module: {len(placeholders)}, reference_module: {len(reference_placeholders)}')
    name_to_node: Dict[str, torch.fx.Node] = {}
    for node in module.graph.nodes:
        name_to_node[node.name] = node
    for placeholder, reference_placeholder in zip(placeholders, reference_placeholders):
        placeholder.target = reference_placeholder.target
        set_node_name(placeholder, reference_placeholder.name, name_to_node)
    module.recompile()