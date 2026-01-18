from collections import defaultdict
from typing import Any, Callable, DefaultDict, Dict, Optional, Tuple, Type
import torch
import torch.fx
from torch.utils import _pytree as pytree
from torch.utils._pytree import tree_map
from .virtualized import V
def matches_module_function_pattern(pattern: Tuple[Type[torch.nn.modules.Module], Callable[..., Any]], node: torch.fx.node.Node, modules: Dict[str, torch.nn.modules.Module]) -> bool:
    if len(node.args) == 0:
        return False
    if not isinstance(node.args[0], torch.fx.Node) or not isinstance(node, torch.fx.Node):
        return False
    if node.args[0].op != 'call_module':
        return False
    if not isinstance(node.args[0].target, str):
        return False
    if node.args[0].target not in modules:
        return False
    if type(modules[node.args[0].target]) is not pattern[0]:
        return False
    if node.op != 'call_function' and node.op != 'call_method':
        return False
    if node.target != pattern[1]:
        return False
    if len(node.args[0].users) > 1:
        return False
    return True