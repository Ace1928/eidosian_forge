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
def mkldnn_one_layer_lstm(inp, hidden, params, has_biases, reverse=False):
    w0 = params[0]
    w1 = params[1]
    if has_biases:
        w2 = params[2]
        w3 = params[3]
    else:
        w2 = torch.zeros(w0.size())
        w3 = torch.zeros(w1.size())
    hx = hidden[0].unsqueeze(0)
    cx = hidden[1].unsqueeze(0)
    batch_sizes: List[int] = []
    mode = 2
    hidden_size = hx.size(2)
    num_layers = 1
    bidirectional = False
    batch_first = False
    train = False
    inp = inp.contiguous()
    hx = hx.contiguous()
    cx = cx.contiguous()
    outputs = torch.ops.aten.mkldnn_rnn_layer.default(inp, w0, w1, w2, w3, hx, cx, reverse, batch_sizes, mode, hidden_size, num_layers, has_biases, bidirectional, batch_first, train)
    y, hy, cy = (outputs[0], outputs[1], outputs[2])
    return (y, (hy.squeeze(0), cy.squeeze(0)))