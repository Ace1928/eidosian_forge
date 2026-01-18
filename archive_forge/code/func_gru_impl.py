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
@register_decomposition(aten.gru.input)
@aten.gru.input.py_impl(DispatchKey.CompositeImplicitAutograd)
@aten.gru.input.py_impl(DispatchKey.Autograd)
def gru_impl(input, hx, params, has_biases, num_layers, dropout, train, bidirectional, batch_first):
    params = gather_params(params, has_biases, False)
    out, final_hiddens = _rnn_helper(input, hx.unbind(0), params, has_biases, num_layers, dropout, train, bidirectional, batch_first, partial(one_layer_rnn, hidden_fn=gru_cell))
    return (out, torch.stack(final_hiddens, 0))