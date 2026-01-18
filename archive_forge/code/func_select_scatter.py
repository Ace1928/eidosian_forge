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
@register_lowering(aten.select_scatter, type_promotion_kind=None)
def select_scatter(x, src, dim: int, index: int):
    assert x.get_dtype() == src.get_dtype()
    x_loader = x.make_loader()
    dim = _validate_dim(x, dim, 0)
    if V.graph.sizevars.evaluate_expr(sympy.Lt(index, 0)):
        index = index + x.get_size()[dim]
    V.graph.sizevars.guard_leq(0, index)
    V.graph.sizevars.guard_lt(index, x.get_size()[dim])
    src = expand(unsqueeze(src, dim), x.get_size())
    src_loader = src.make_loader()

    def inner_fn(idx):
        return ops.where(ops.eq(ops.index_expr(idx[dim], torch.int32), ops.index_expr(index, torch.int32)), src_loader(idx), x_loader(idx))
    return Pointwise.create(device=x.get_device(), dtype=x.get_dtype(), inner_fn=inner_fn, ranges=list(x.get_size()))