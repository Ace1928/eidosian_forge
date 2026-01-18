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
def gru_cell(inp, cur_hidden, ih_weight, ih_bias, hh_weight, hh_bias):
    chunked_igates = inp.chunk(3, 1)
    chunked_hgates = F.linear(cur_hidden, hh_weight, hh_bias).chunk(3, 2)
    reset_gate = (chunked_hgates[0] + chunked_igates[0]).sigmoid()
    input_gate = (chunked_hgates[1] + chunked_igates[1]).sigmoid()
    new_gate = (chunked_igates[2] + chunked_hgates[2] * reset_gate).tanh()
    return (cur_hidden - new_gate) * input_gate + new_gate