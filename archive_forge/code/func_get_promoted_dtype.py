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
def get_promoted_dtype(*args, type_promotion_kind: ELEMENTWISE_TYPE_PROMOTION_KIND):

    def construct_input(inp):
        if isinstance(inp, (Number, sympy.Symbol)):
            return inp
        else:
            assert hasattr(inp, 'get_dtype')
            dim = len(inp.get_size())
            return torch.zeros([1] * dim, dtype=inp.get_dtype())
    inps = [construct_input(arg) for arg in args]
    _, dtype = elementwise_dtypes(*inps, type_promotion_kind=type_promotion_kind)
    return dtype