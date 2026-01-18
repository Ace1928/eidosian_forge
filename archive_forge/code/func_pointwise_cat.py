import functools
import itertools
import logging
import os
import warnings
from collections import defaultdict
from collections.abc import Iterable
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import sympy
import torch
import torch.fx
import torch.utils._pytree as pytree
from torch._higher_order_ops.triton_kernel_wrap import (
from torch._prims_common import (
from torch.fx.experimental.sym_node import magic_methods, method_to_operator
from torch.utils._sympy.functions import CeilDiv, FloorDiv, ModularIndexing
from .._dynamo.utils import import_submodule
from . import config, inductor_prims, ir, test_operators  # NOQA: F401
from .decomposition import decompositions, get_decompositions
from .ir import (
from .utils import (
from .virtualized import ops, V
from . import kernel
import_submodule(kernel)
from . import quantized_lowerings
def pointwise_cat(inputs, dim=0):
    inputs_ranges: List[Tuple[sympy.Expr, sympy.Expr]] = []
    prev_end = 0
    for inp in inputs:
        inputs_ranges.append((prev_end, prev_end + inp.get_size()[dim]))
        prev_end = inputs_ranges[-1][-1]
    inputs_loaders = [inp.make_loader() for inp in inputs]

    def inner_fn(idx):
        idx_dim = ops.index_expr(idx[dim], torch.int64)
        masks = []
        masked_loads = []
        for i in range(len(inputs)):
            start = ops.constant(0, torch.int64) if i == 0 else ops.index_expr(inputs_ranges[i][0], torch.int64)
            end = ops.index_expr(inputs_ranges[i][1], torch.int64)
            start_cond = ops.ge(idx_dim, start)
            end_cond = ops.lt(idx_dim, end)
            if i == 0:
                mask = end_cond
            elif i == len(inputs) - 1:
                mask = start_cond
            else:
                mask = ops.and_(start_cond, end_cond)
            masks.append(mask)
            idx_load = list(idx)
            idx_load[dim] -= inputs_ranges[i][0]
            masked_loads.append(ops.masked(mask, lambda: inputs_loaders[i](idx_load), 0.0))

        def get_masked_val(i):
            if i != len(inputs) - 1:
                return ops.where(masks[i], masked_loads[i], get_masked_val(i + 1))
            else:
                return masked_loads[-1]
        return get_masked_val(0)
    new_size = list(inputs[0].get_size())
    new_size[dim] = inputs_ranges[-1][-1]
    return Pointwise.create(device=inputs[0].get_device(), dtype=inputs[0].get_dtype(), inner_fn=inner_fn, ranges=new_size)