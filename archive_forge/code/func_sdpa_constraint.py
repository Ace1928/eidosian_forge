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
def sdpa_constraint(fx_node, *args, **kwargs):

    def apply_constraint(arg, fx_arg):
        if not isinstance(arg, ir.IRNode):
            return arg
        meta_val = fx_arg.meta['val']
        if not meta_val.is_cuda:
            return arg
        stride_order = ir.get_stride_order(meta_val.stride())
        if stride_order and stride_order[-1] != 0:
            stride_order = list(reversed(range(len(arg.get_size()))))
        ALIGNMENT = 8
        is_backward = fx_node.target in (aten._scaled_dot_product_efficient_attention_backward.default, aten._scaled_dot_product_flash_attention_backward.default)

        def is_aligned(x):
            return V.graph.sizevars.size_hint(x.get_size()[-1]) % ALIGNMENT == 0
        assert isinstance(arg, TensorBox)
        if isinstance(arg.data, (ir.SliceView, ir.ExpandView)):
            if not is_aligned(arg):
                if is_aligned(arg.unwrap_view()):
                    return arg

        def is_aligned_backward(x):
            aligned_strides = all((V.graph.sizevars.size_hint(x.get_stride()[i]) % ALIGNMENT == 0 for i in range(len(x.get_stride()) - 1)))
            return V.graph.sizevars.size_hint(x.get_stride()[-1]) == 1 and aligned_strides
        if isinstance(arg.data, ir.StorageBox) and arg.data.is_input_buffer() and is_backward:
            if len(arg.data.get_size()) == 4 and is_aligned_backward(arg):
                return arg
        return ir.ExternKernel.require_stride_order(arg, stride_order)
    args = tuple((apply_constraint(arg, fx_arg) for arg, fx_arg in zip(args, fx_node.args)))
    kwargs = {k: apply_constraint(v, fx_node.kwargs[k]) for k, v in kwargs.items()}
    return (args, kwargs)