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
@register_lowering(aten.avg_pool2d_backward, type_promotion_kind=None)
def avg_pool2d_backward(grad_output, x, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override=None):
    assert divisor_override is None or divisor_override != 0, 'divisor must be not zero'
    if not stride:
        stride = kernel_size
    if not padding:
        padding = [0, 0]
    assert isinstance(grad_output, TensorBox)
    assert isinstance(x, TensorBox)
    assert len(kernel_size) == 2
    assert len(stride) == 2
    assert len(padding) == 2
    assert len(x.get_size()) in (3, 4)
    grad_output.realize_hint()
    *batch, height, width = x.get_size()
    h_out, ceil_mode1 = pooling_size(height, 0, kernel_size, stride, padding, ceil_mode)
    w_out, ceil_mode2 = pooling_size(width, 1, kernel_size, stride, padding, ceil_mode)
    grad_loader = grad_output.make_loader()
    had_padding = padding[0] or padding[1] or ceil_mode1 or ceil_mode2
    *_, pooled_height, pooled_width = grad_output.get_size()
    new_size = list(x.get_size())
    dtype = x.get_dtype()
    h_window_size = max([max(h // stride[0] - max(0, (h - kernel_size[0]) // stride[0]), 1) for h in range(kernel_size[0] * 2)])
    w_window_size = max([max(w // stride[1] - max(0, (w - kernel_size[1]) // stride[1]), 1) for w in range(kernel_size[1] * 2)])
    window_size = h_window_size * w_window_size
    if window_size > 25:
        return fallback_avg_pool2d_backward(grad_output, x, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override)

    def compute_pool_size_without_padding(ph, pw):
        """
        This computes the scaling factor that we will divide an element
        by when `count_include_pad=False`
        """
        stride_h = ops.constant(stride[0], torch.int32)
        stride_w = ops.constant(stride[1], torch.int32)
        pad_h = ops.constant(padding[0], torch.int32)
        pad_w = ops.constant(padding[1], torch.int32)
        kernel_h = ops.constant(kernel_size[0], torch.int32)
        kernel_w = ops.constant(kernel_size[1], torch.int32)
        hstart = ops.sub(ops.mul(ph, stride_h), pad_h)
        wstart = ops.sub(ops.mul(pw, stride_w), pad_w)
        hend = ops.minimum(ops.add(hstart, kernel_h), ops.add(ops.index_expr(height, torch.int32), pad_h))
        wend = ops.minimum(ops.add(wstart, kernel_w), ops.add(ops.index_expr(width, torch.int32), pad_w))
        hstart = ops.maximum(hstart, ops.constant(0, torch.int32))
        wstart = ops.maximum(wstart, ops.constant(0, torch.int32))
        hend = ops.minimum(hend, ops.index_expr(height, torch.int32))
        wend = ops.minimum(wend, ops.index_expr(width, torch.int32))
        divide_factor = ops.mul(ops.sub(hend, hstart), ops.sub(wend, wstart))
        return divide_factor

    def fn(idx):
        *prefix, h, w = idx
        h = h + padding[0]
        w = w + padding[1]
        phstart = ops.index_expr(FloorDiv(h - kernel_size[0] + stride[0], stride[0]), torch.int32)
        pwstart = ops.index_expr(FloorDiv(w - kernel_size[1] + stride[1], stride[1]), torch.int32)
        phend = ops.index_expr(FloorDiv(h, stride[0]) + 1, torch.int32)
        pwend = ops.index_expr(FloorDiv(w, stride[1]) + 1, torch.int32)
        phstart = ops.maximum(phstart, ops.constant(0, torch.int32))
        pwstart = ops.maximum(pwstart, ops.constant(0, torch.int32))
        phend = ops.minimum(phend, ops.index_expr(pooled_height, torch.int32))
        pwend = ops.minimum(pwend, ops.index_expr(pooled_width, torch.int32))
        gradient = None
        for ph_ in range(h_window_size):
            for pw_ in range(w_window_size):
                ph = ops.add(phstart, ops.constant(ph_, torch.int32))
                pw = ops.add(pwstart, ops.constant(pw_, torch.int32))
                if divisor_override is not None:
                    scale = divisor_override
                elif count_include_pad or not had_padding:
                    scale = kernel_size[0] * kernel_size[1]
                else:
                    scale = compute_pool_size_without_padding(ph, pw)
                part = ops.truediv(grad_loader([*prefix, ops.indirect_indexing(ops.minimum(ph, ops.sub(phend, ops.constant(1, torch.int32))), pooled_height, check=False), ops.indirect_indexing(ops.minimum(pw, ops.sub(pwend, ops.constant(1, torch.int32))), pooled_width, check=False)]), scale)
                mask = ops.and_(ops.lt(ph, phend), ops.lt(pw, pwend))
                if gradient is None:
                    gradient = ops.where(mask, part, ops.constant(0.0, torch.float32))
                else:
                    gradient = ops.where(mask, ops.add(gradient, part), gradient)
        assert gradient is not None
        return gradient
    rv = Pointwise.create(device=grad_output.get_device(), dtype=dtype, inner_fn=fn, ranges=new_size)
    return rv