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
@register_decomposition(aten.lstm.data)
@aten.lstm.data.py_impl(DispatchKey.CompositeImplicitAutograd)
@aten.lstm.data.py_impl(DispatchKey.Autograd)
def lstm_data_impl(data, batch_sizes, hx, params, has_biases, num_layers, dropout, train, bidirectional):
    assert len(hx) == 2, 'lstm expects two hidden states'
    params = gather_params(params, has_biases, hx[0].size(2) != hx[1].size(2))
    hidden = list(zip(hx[0], hx[1]))
    out, final_hiddens = _rnn_helper(data, hidden, params, has_biases, num_layers, dropout, train, bidirectional, False, partial(one_layer_lstm_data, batch_sizes=batch_sizes))
    final_hiddens = list(zip(*final_hiddens))
    return (out, torch.stack(final_hiddens[0], 0), torch.stack(final_hiddens[1], 0))