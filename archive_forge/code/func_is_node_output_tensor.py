from typing import List, Tuple, Union, Dict, Any, Set, Mapping
import collections
from dataclasses import dataclass
import torch
import torch.fx
from torch.fx.node import _get_qualified_name
from torch.fx._compatibility import compatibility
@compatibility(is_backward_compatible=False)
def is_node_output_tensor(node: torch.fx.Node) -> bool:
    """Checks if the node output produces a Tensor or not.

    NOTE: This requires to run `ShapeProp` on the containing fx graph before
    calling this function. This is because it works by checking the `type`
    metadata on the node. This metadata is produced by the `ShapeProp`.
    """
    type_ = node.meta.get('type', None)
    return type_ is not None and issubclass(type_, torch.Tensor)