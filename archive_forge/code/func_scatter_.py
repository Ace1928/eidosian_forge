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
@register_lowering(aten.scatter_, type_promotion_kind=None)
def scatter_(self, dim: int, index, src, *, reduce: Optional[str]=None):
    assert reduce in {None, 'add', 'multiply'}
    fallback_result = scatter_fallback('aten.scatter_', self, dim, index, src, reduce=reduce)
    if fallback_result:
        return fallback_result
    if reduce == 'add':
        reduce = 'sum'
    elif reduce == 'multiply':
        reduce = 'prod'
    return scatter_reduce_(self, dim, index, src, reduce)