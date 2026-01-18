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
def maybe_dequantize_first_two_tensor_args_and_handle_tuples(f):

    def inner(*args, **kwargs):
        a0, a1, *a_other = args
        if isinstance(a0, tuple) and isinstance(a1, tuple) or (isinstance(a0, list) and isinstance(a1, list)):
            results = []
            for el0, el1 in zip(a0, a1):
                new_args = (el0, el1, *a_other)
                results.append(inner(*new_args, **kwargs))
            return results
        elif isinstance(a0, torch.Tensor) and isinstance(a1, torch.Tensor):
            if a0.is_quantized:
                a0 = a0.dequantize()
            if a1.is_quantized:
                a1 = a1.dequantize()
        if a0.dtype != torch.float or a1.dtype != torch.float:
            return None
        new_args = (a0, a1, *a_other)
        return f(*new_args, **kwargs)
    return inner