import math
from enum import Enum
from functools import partial
from typing import List, Optional, Sequence, Tuple, Union
import torch
import torch._prims_common as utils
from torch import SymBool, SymFloat, Tensor
from torch._decomp import (
from torch._ops import OpOverload
from torch._prims import _prim_elementwise_meta, ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND
from torch._prims_common import (
from torch._prims_common.wrappers import (
from torch._refs import _broadcast_shapes, _maybe_broadcast
from torch.utils import _pytree as pytree
import torch._refs
import torch._refs.nn.functional
import torch._refs.special
def rnn_cell_checkSizes(input_gates, hidden_gates, input_bias, hidden_bias, factor, prev_hidden):
    torch._check(input_gates.ndim == 2, lambda: f'{input_gates.ndim} != 2')
    torch._check(input_gates.shape == hidden_gates.shape, lambda: f'{input_gates.shape} != {hidden_gates.shape}')
    gates_size = input_gates.size(1)
    if input_bias is not None:
        torch._check(input_bias.ndim == 1, lambda: f'{input_bias.ndim} != 1')
        torch._check(input_bias.numel() == gates_size, lambda: f'{input_bias.numel()} != {gates_size}')
        torch._check(input_bias.shape == hidden_bias.shape, lambda: f'{input_bias.shape} != {hidden_bias.shape}')
    torch._check(prev_hidden.ndim == 2, lambda: f'{prev_hidden.ndim} != 2')
    expected_prev_hidden_numel = input_gates.size(0) * gates_size // factor
    torch._check(prev_hidden.numel() == expected_prev_hidden_numel, lambda: f'{prev_hidden.numel()} != {input_gates.size(0)} * {gates_size} // {factor} (aka {expected_prev_hidden_numel})')
    torch._check(all((x.device == input_gates.device for x in [hidden_gates, input_bias, hidden_bias, prev_hidden])), lambda: 'expected all inputs to be same device')