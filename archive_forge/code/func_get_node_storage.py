from collections import defaultdict
from typing import Any, Callable, DefaultDict, Dict, Optional, Tuple, Type
import torch
import torch.fx
from torch.utils import _pytree as pytree
from torch.utils._pytree import tree_map
from .virtualized import V
def get_node_storage(node: torch.fx.Node) -> Optional[int]:
    if 'val' not in node.meta:
        return None
    if not isinstance(node.meta['val'], torch.Tensor):
        return None
    if not torch._C._has_storage(node.meta['val']):
        return None
    return get_storage(node.meta['val'])