import enum
import operator
import torch
import torch.nn as nn
import torch.ao.nn.intrinsic.quantized as nniq
import torch.ao.nn.quantized as nnq
from typing import Tuple, Callable, Dict, Set, List, Optional, Union
from torch.fx import GraphModule
from torch.fx.graph import Node
from torch.ao.quantization import (
from torch.ao.quantization.utils import getattr_from_fqn
from torch.ao.quantization.observer import _is_activation_post_process
from .ns_types import NSNodeTargetType, NSResultsType
def get_arg_indices_of_inputs_to_log(node: Node) -> List[int]:
    """
    Returns the indices of args of the node which we should attach
    loggers to, if input logging is enabled.

    For example,
    * for (x + y), returns [0, 1]
    * for (1 + y), returns [1]
    * for (x + 1), returns [0]
    * for (linear(x, w, b)) returns [0]
    * by default, returns [0]
    """
    if len(node.args) == 0:
        return []
    if node.op == 'call_function' and (node.target in (torch.add, torch.ops.quantized.add, operator.add) or node.target in (torch.mul, torch.ops.quantized.mul, operator.mul)):
        result = []
        for i in range(2):
            if type(node.args[i]) == Node:
                result.append(i)
        return result
    return [0]