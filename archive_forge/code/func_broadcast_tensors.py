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
@register_lowering(aten.broadcast_tensors, broadcast=False, type_promotion_kind=None)
def broadcast_tensors(*inputs):
    if len(inputs) == 1 and isinstance(inputs[0], (list, tuple)):
        return broadcast_tensors(*inputs[0])
    target: List[sympy.Expr] = functools.reduce(broadcast_symbolic_shapes, [x.get_size() for x in inputs], [])
    outputs = []
    for x in inputs:
        sizes = x.get_size()
        if len(sizes) != len(target) or any((a == 1 and b != 1 or (a != 1 and b == 1) for a, b in zip(sizes, target))):
            x = expand(x, target)
        outputs.append(x)
    return outputs