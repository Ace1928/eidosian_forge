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
def index_put_fallback(self, indices, values, accumulate):
    deterministic = torch.are_deterministic_algorithms_enabled()
    if is_triton(values) and (accumulate or deterministic):
        V.graph.disable_cudagraphs = True
        msg = 'index put with accumulate.' if not deterministic else 'deterministic index put.'
        if (stack_trace := V.graph.current_node.meta.get('stack_trace', None)):
            msg = f'{msg} Found from : \n {stack_trace}'
        V.graph.disable_cudagraphs_reason = msg
    ir.IndexPutFallback(self, indices, values, accumulate)
    return self