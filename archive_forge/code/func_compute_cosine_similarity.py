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
@maybe_dequantize_first_two_tensor_args_and_handle_tuples
def compute_cosine_similarity(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Computes the cosine similarity between `x` and `y`.

    Args:
        x: Tensor or tuple of tensors
        y: Tensor or tuple of tensors

    Return:
        float or tuple of floats
    """
    x = x.reshape(1, -1)
    y = y.reshape(1, -1)
    return torch.nn.functional.cosine_similarity(x, y)