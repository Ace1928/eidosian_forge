from torch.fx.graph_module import GraphModule
from typing import Any, Callable, Dict, List, Tuple, Type
import torch
import torch.nn as nn
from torch.fx._compatibility import compatibility
@compatibility(is_backward_compatible=False)
def default_matching(name: str, target_version: int) -> str:
    """Default matching method
    """
    return name