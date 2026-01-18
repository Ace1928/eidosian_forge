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
def pooling_size(x, i, kernel_size, stride, padding, ceil_mode):
    x_out = FloorDiv(x + 2 * padding[i] - (kernel_size[i] - 1) + (stride[i] - 1), stride[i])
    if ceil_mode:
        x_alt = FloorDiv(x + 2 * padding[i] - (kernel_size[i] - 1) + 2 * (stride[i] - 1), stride[i])
        if V.graph.sizevars.size_hint((x_alt - 1) * stride[i] - x - padding[i]) >= 0:
            x_alt -= 1
            V.graph.sizevars.guard_leq(0, x_alt * stride[i] - x - padding[i])
        if V.graph.sizevars.size_hint(x_out - x_alt) == 0:
            V.graph.sizevars.guard_equals(x_out, x_alt)
            ceil_mode = False
        else:
            x_out = x_alt
    return (x_out, ceil_mode)