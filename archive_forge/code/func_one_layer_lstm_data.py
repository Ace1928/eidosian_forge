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
def one_layer_lstm_data(inp, hidden, params, has_biases, batch_sizes, reverse=False):
    ih_weight = params[0]
    hh_weight = params[1]
    ih_bias = params[2] if has_biases else None
    hh_bias = params[3] if has_biases else None
    hr_weight = params[4] if len(params) == 5 else params[2] if len(params) == 3 else None
    step_output = []
    hiddens = []
    last_batch_size = batch_sizes[-1] if reverse else batch_sizes[0]
    split_inp = torch.split(inp, list(batch_sizes))
    if reverse:
        split_inp = split_inp[::-1]
    orig_hx = hidden[0]
    orig_cx = hidden[1]
    hx, cx = (orig_hx.narrow(0, 0, last_batch_size), orig_cx.narrow(0, 0, last_batch_size))
    for inp in split_inp:
        i = inp.shape[0]
        inp = F.linear(inp, ih_weight, ih_bias)
        if i < last_batch_size:
            hiddens.append((hx.narrow(0, i, last_batch_size - i), cx.narrow(0, i, last_batch_size - i)))
            hx, cx = (hx.narrow(0, 0, i), cx.narrow(0, 0, i))
        if i > last_batch_size:
            hx = torch.concat((hx, orig_hx.narrow(0, last_batch_size, i - last_batch_size)), 0)
            cx = torch.concat((cx, orig_cx.narrow(0, last_batch_size, i - last_batch_size)), 0)
        hx, cx = lstm_cell(inp, hx, cx, hh_weight, hh_bias, hr_weight, chunk_dim=1)
        last_batch_size = i
        step_output.append(hx)
    if reverse:
        step_output.reverse()
        hidden_out = (hx, cx)
    else:
        hiddens.append((hx, cx))
        hiddens.reverse()
        hidden0, hidden1 = zip(*hiddens)
        hidden_out = (torch.cat(hidden0, 0), torch.cat(hidden1, 0))
    out = torch.cat(step_output, 0)
    return (out, hidden_out)