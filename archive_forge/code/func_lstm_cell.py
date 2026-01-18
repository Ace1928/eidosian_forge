import functools
import numbers
import operator
import sys
from enum import Enum
from functools import partial, reduce
from itertools import chain, product
from typing import Callable, cast, Iterable, List, Optional, Tuple, Union
import torch
import torch._prims as prims
import torch._prims_common as utils
import torch.nn.functional as F
from torch import sym_float, sym_int, Tensor
from torch._decomp import register_decomposition
from torch._higher_order_ops.out_dtype import out_dtype
from torch._prims_common import IntLike, NumberType, TensorLike, TensorSequenceType
from torch._prims_common.wrappers import (
from torch.utils import _pytree as pytree
from torch.utils._pytree import tree_map
def lstm_cell(inp, hx, cx, hh_weight, hh_bias, hr_weight, chunk_dim):
    gates = F.linear(hx, hh_weight, hh_bias) + inp
    chunked_gates = gates.chunk(4, chunk_dim)
    in_gate = chunked_gates[0].sigmoid()
    forget_gate = chunked_gates[1].sigmoid()
    cell_gate = chunked_gates[2].tanh()
    out_gate = chunked_gates[3].sigmoid()
    cy = forget_gate * cx + in_gate * cell_gate
    hy = out_gate * cy.tanh()
    hy = hy if hr_weight is None else F.linear(hy, hr_weight, None)
    return (hy, cy)