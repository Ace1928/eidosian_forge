import copy
import torch
import torch.nn as nn
from torch.ao.quantization import (
from torch.ao.quantization.backend_config import (
from torch.ao.quantization.fake_quantize import (
from torch.ao.quantization.observer import (
from torch.ao.quantization.qconfig import (
from torch.ao.quantization.stubs import DeQuantStub
from torch.ao.quantization.utils import (
from torch.ao.quantization.observer import _is_activation_post_process
from torch.ao.quantization.qconfig_mapping import QConfigMapping
from torch.fx import GraphModule, map_arg
from torch.fx.graph import (
from .custom_config import PrepareCustomConfig
from ._decomposed import quantized_decomposed_lib  # noqa: F401
from typing import Callable, Optional, List, Dict, Any, Set, Tuple, Union, Type
from dataclasses import dataclass
from collections import namedtuple
import operator
import warnings
def maybe_get_next_module(node: Node, modules: Dict[str, nn.Module], target_module_type: Optional[Type[nn.Module]]=None, target_functional_type: Any=None) -> Optional[Node]:
    """ Gets the next module that matches what is needed in
    is_target_module_type if it exists

    Args:
        node: The node whose users we want to look at
        target_module_type: Module type that we want to check
        target_functional_type: Functional type that we want to check
    """
    for user in node.users.keys():
        if user.op == 'call_module' and target_module_type is not None and isinstance(modules[str(user.target)], target_module_type):
            return user
        elif user.op == 'call_function' and target_functional_type is not None and (user.target == target_functional_type):
            return user
    return None