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
def scatter_fallback(fn, self, dim: int, index, src, *, reduce: Optional[str]=None, include_self: bool=True):
    reduce_ty = 'add' if fn == 'aten.scatter_' else 'sum'
    if reduce not in {None, reduce_ty} or (isinstance(src, TensorBox) and src.get_device().type == torch.device('cuda').type and needs_fallback_due_to_atomic_add_limitations(src.get_dtype())) or (fn == 'aten.scatter_reduce_' and reduce == 'sum' and isinstance(src, TensorBox) and (src.get_device() == torch.device('cpu')) and config.cpp.fallback_scatter_reduce_sum) or (reduce == reduce_ty and self.get_dtype() in {torch.bool, torch.int64}) or torch.are_deterministic_algorithms_enabled():
        ir.ScatterFallback(fn, self, dim, index, src, reduce=reduce, include_self=include_self)
        return self
    return None