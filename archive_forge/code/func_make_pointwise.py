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
def make_pointwise(fn, override_return_dtype=None, override_device=None, override_fn_when_input_bool=None, override_fn_when_cuda_float64=None, allow_alpha=False):

    def inner(*inputs: List[TensorBox], alpha=None):
        inputs = promote_constants(inputs, override_return_dtype)
        if allow_alpha:
            if alpha is not None and alpha != 1:
                inputs = list(inputs)
                inputs[-1] = mul(inputs[-1], alpha)
        else:
            assert alpha is None
        loaders = [x.make_loader() for x in inputs]
        ranges = inputs[0].get_size()
        dtype = override_return_dtype or inputs[0].get_dtype()
        is_cuda = decode_device(inputs[0].get_device()).type == 'cuda'
        for other in inputs[1:]:
            assert isinstance(other, ir.BaseConstant) or len(ranges) == len(other.get_size()), f'ndim mismatch {fn} {ranges} {other.get_size()}'

        def inner_fn(index):
            assert len(index) == len(ranges), f'wrong ndim {index} {ranges}'
            if dtype == torch.bool and override_fn_when_input_bool is not None:
                return override_fn_when_input_bool(*[load(index) for load in loaders])
            elif override_fn_when_cuda_float64 and is_cuda and (dtype == torch.float64):
                return override_fn_when_cuda_float64(*[load(index) for load in loaders])
            else:
                return fn(*[load(index) for load in loaders])
        if not override_device:
            device = None
            for i in inputs:
                if i.get_device().type == 'cuda':
                    device = i.get_device()
                    break
            if not device:
                device = inputs[0].get_device()
        device = override_device or device
        return Pointwise.create(device=device, dtype=dtype, inner_fn=inner_fn, ranges=ranges)
    return inner