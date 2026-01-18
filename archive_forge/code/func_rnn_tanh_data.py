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
@register_decomposition(aten.rnn_tanh.data)
@aten.rnn_tanh.data.py_impl(DispatchKey.CompositeImplicitAutograd)
@aten.rnn_tanh.data.py_impl(DispatchKey.Autograd)
def rnn_tanh_data(data, batch_sizes, hx, params, has_biases, num_layers, dropout, train, bidirectional):
    hidden = hx.unbind(0)
    params = gather_params(params, has_biases, False)
    out, final_hiddens = _rnn_helper(data, hidden, params, has_biases, num_layers, dropout, train, bidirectional, False, partial(one_layer_rnn_data, batch_sizes=batch_sizes, hidden_fn=rnn_cell_data(torch.tanh)))
    return (out, torch.stack(final_hiddens, 0))