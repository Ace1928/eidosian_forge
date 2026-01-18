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
def get_number_of_non_param_args(node: Node, gm: GraphModule) -> int:
    """
    Assumes that all non-param args occur first. Returns the number of
    non-param args expected for a node.  For example, for

      F.linear(x, weight, bias)

    Returns 1, because x is a non-param arg and weight and bias are params.
    For

      lstm_mod(x, hid)

    Returns 2, because both x and hid are non-param args.
    """
    if node.op == 'call_module':
        node_obj = getattr_from_fqn(gm, node.target)
        if isinstance(node_obj, nn.LSTM):
            return 2
    return 1