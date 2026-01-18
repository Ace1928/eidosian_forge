import sys
import torch
from torch.fx.graph import (
from torch.ao.quantization.utils import Pattern
from .quantize_handler import (
from ..qconfig import (
from ..utils import (
from .graph_module import (
from torch.nn.utils.parametrize import type_before_parametrizations
from typing import Any, Dict, List, Callable, Optional, Tuple, Type, Set, Iterable
def is_standalone_module(node_target: str, modules: Dict[str, torch.nn.Module]):
    assert modules is not None
    return node_target in standalone_module_names or type(modules[node_target]) in standalone_module_classes