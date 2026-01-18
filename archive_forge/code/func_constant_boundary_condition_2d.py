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
def constant_boundary_condition_2d(x, fill_value, padding=None, pad_fill_value=1.0):
    *_, h, w = x.get_size()
    x_loader = x.make_loader()
    padding_h = padding[0] if padding else 0
    padding_w = padding[1] if padding else 0

    def load(index):
        *prefix, ih, iw = index
        mask = ops.and_(range_mask(ih, h + padding_h, -padding_h), range_mask(iw, w + padding_w, -padding_w))
        return ops.masked(mask, lambda: constant_boundary_condition_2d(x, pad_fill_value)([*prefix, ih, iw]), fill_value) if padding else ops.masked(mask, lambda: x_loader([*prefix, ih, iw]), fill_value)
    return load