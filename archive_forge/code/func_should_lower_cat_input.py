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
def should_lower_cat_input(x) -> bool:
    if ir.is_storage_and_layout(x):
        storage, _ = ir.as_storage_and_layout(x, freeze=False)
        return not ir.ConcatKernel.can_realize_into_without_copy(storage)
    if isinstance(x, TensorBox):
        if isinstance(x.data, ir.BaseView):
            return should_lower_cat_input(x.data.unwrap_view())
        else:
            return should_lower_cat_input(x.data)
    if isinstance(x, ir.StorageBox):
        return should_lower_cat_input(x.data)
    if isinstance(x, ir.Pointwise):
        return True
    return False