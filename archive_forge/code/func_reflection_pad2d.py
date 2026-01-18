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
@register_lowering(aten.reflection_pad2d)
def reflection_pad2d(x, padding):
    assert len(padding) == 4
    left, right, top, bot = padding
    x_loader = x.make_loader()
    *batch, h, w = x.get_size()

    def reflect(x, size, offset):
        size_num = size
        size = ops.index_expr(size - 1, torch.int32)
        x = ops.index_expr(x, torch.int32)
        x = ops.sub(x, ops.index_expr(offset, torch.int32))
        x = ops.sub(size, ops.abs(ops.sub(size, ops.abs(x))))
        return ops.indirect_indexing(x, size_num, check=False)

    def fn(idx):
        *b, x, y = idx
        x = reflect(x, h, top)
        y = reflect(y, w, left)
        return x_loader([*b, x, y])
    return Pointwise.create(device=x.get_device(), dtype=x.get_dtype(), inner_fn=fn, ranges=[*batch, h + top + bot, w + left + right])