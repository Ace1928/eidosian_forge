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
def all_node_args_have_no_tensors(node: Node, modules: Dict[str, torch.nn.Module], cache: Dict[Node, bool]) -> bool:
    """
    If we know for sure that all of this node's args have no
    tensors (are primitives), return True.  If we either
    find a tensor or are not sure, return False. Note: this
    function is not exact.
    """
    if cache and node in cache:
        return cache[node]
    result = False
    if not isinstance(node, Node):
        result = True
    elif node.op == 'placeholder':
        result = False
    elif node.op == 'call_module':
        assert isinstance(node.target, str)
        if _is_activation_post_process(modules[node.target]):
            result = all_node_args_have_no_tensors(node.args[0], modules, cache)
    elif node.op == 'call_module':
        result = False
    elif node.op == 'call_function' and node.target is operator.getitem:
        result = all_node_args_have_no_tensors(node.args[0], modules, cache)
    elif node.op == 'get_attr':
        result = False
    elif node.target is getattr and node.args[1] in ['ndim', 'shape']:
        result = True
    elif node.op == 'call_method' and node.target == 'size':
        result = True
    else:
        found_one_tensor = False
        for arg in node.args:
            if isinstance(arg, list):
                for list_el in arg:
                    if isinstance(list_el, Node):
                        this_list_el_args_have_no_tensors = all_node_args_have_no_tensors(list_el, modules, cache)
                        found_one_tensor = found_one_tensor or not this_list_el_args_have_no_tensors
                        if found_one_tensor:
                            result = not found_one_tensor
                            if cache:
                                cache[node] = result
                            return result
            elif isinstance(arg, int):
                pass
            elif isinstance(arg, Node):
                this_arg_args_have_no_tensors = all_node_args_have_no_tensors(arg, modules, cache)
                found_one_tensor = found_one_tensor or not this_arg_args_have_no_tensors
                if found_one_tensor:
                    result = not found_one_tensor
                    if cache:
                        cache[node] = result
                    return result
            else:
                found_one_tensor = True
            result = not found_one_tensor
    if cache:
        cache[node] = result
    return result