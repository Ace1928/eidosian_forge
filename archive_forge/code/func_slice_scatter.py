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
@register_lowering(aten.slice_scatter, type_promotion_kind=None)
def slice_scatter(x, src, dim=0, start=None, end=None, step=1):
    assert x.get_dtype() == src.get_dtype()
    x_loader = x.make_loader()
    dim = _validate_dim(x, dim, 0)
    dim_size = x.get_size()[dim]
    if start is not None and V.graph.sizevars.evaluate_expr(sympy.Lt(start, 0)):
        start = start + dim_size
    if end is not None and V.graph.sizevars.evaluate_expr(sympy.Lt(end, 0)):
        end = end + dim_size
    if start is None:
        start = 0
    if end is None or V.graph.sizevars.statically_known_leq(x.get_size()[dim], end):
        end = dim_size
    src_size = list(x.get_size())
    src_size[dim] = FloorDiv(end - start + (step - 1), step)
    src = expand(src, src_size)
    src_loader = src.make_loader()

    def inner_fn(idx):
        if start == 0 and end == dim_size and (step == 1):
            return src_loader(idx)
        idx_dim = ops.index_expr(idx[dim], torch.int64)
        src_idx = list(idx)
        src_idx[dim] = FloorDiv(idx[dim] - start, step)
        mask = []
        if start != 0:
            mask.append(ops.ge(idx_dim, ops.index_expr(sympy.expand(start), torch.int64)))
        if end != dim_size:
            mask.append(ops.lt(idx_dim, ops.index_expr(sympy.expand(end), torch.int64)))
        if step != 1:
            mask.append(ops.eq(ops.index_expr(ModularIndexing(idx[dim] - start, 1, step), torch.int64), ops.constant(0, torch.torch.int64)))
        assert mask
        mask = functools.reduce(ops.and_, mask)
        src_val = ops.masked(mask, lambda: src_loader(src_idx), 0 if is_integer_type(x) else 0.0)
        return ops.where(mask, src_val, x_loader(idx))
    return Pointwise.create(device=x.get_device(), dtype=x.get_dtype(), inner_fn=inner_fn, ranges=list(x.get_size()))