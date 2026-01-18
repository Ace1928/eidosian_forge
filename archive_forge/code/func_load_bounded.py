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
def load_bounded(fy, fx):
    _0 = ops.constant(0, torch.int32)
    iHm1 = ops.constant(iH - 1, torch.int32)
    iWm1 = ops.constant(iW - 1, torch.int32)
    iy = ops.indirect_indexing(clamp(fy, _0, iHm1), iH, check=False)
    ix = ops.indirect_indexing(clamp(fx, _0, iWm1), iW, check=False)
    return x_loader([n, c, iy, ix])