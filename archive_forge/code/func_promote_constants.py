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
def promote_constants(inputs, override_return_dtype=None):
    if not any((isinstance(x, (sympy.Expr, int, float)) for x in inputs)):
        return inputs
    if all((isinstance(x, (int, float, sympy.Symbol)) for x in inputs)):
        dtype = override_return_dtype or get_promoted_dtype(*inputs, type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT)

        def const_func(x):
            if isinstance(x, sympy.Symbol):
                return ir.IndexingConstant(x, dtype, decode_device(None))
            else:
                return ir.Constant(x, dtype, decode_device(None))
        return [const_func(x) for x in inputs]
    ex = next((x for x in inputs if isinstance(x, (TensorBox, ExpandView))))
    out = []
    for x in inputs:
        if isinstance(x, (int, float)):
            out.append(ExpandView.create(ir.Constant(x, ex.get_dtype(), ex.get_device()), list(ex.get_size())))
        elif isinstance(x, sympy.Expr):
            out.append(ExpandView.create(IndexingConstant(x, ex.get_dtype(), ex.get_device()), list(ex.get_size())))
        else:
            out.append(x)
    return out