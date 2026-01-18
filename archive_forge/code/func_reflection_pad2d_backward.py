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
@register_lowering(aten.reflection_pad2d_backward)
def reflection_pad2d_backward(grad_output, x, padding):
    assert len(padding) == 4
    left, right, top, bot = padding
    *_, h, w = x.get_size()
    h = V.graph.sizevars.evaluate_static_shape(h) - 1
    w = V.graph.sizevars.evaluate_static_shape(w) - 1
    grad_loader = grad_output.make_loader()
    *_, h_grad, w_grad = grad_output.get_size()

    def fn(idx):
        *b, x, y = idx

        def load_from_output(x, y):
            return grad_loader([*b, x, y])

        def index_range_condition(index_range):
            i, lb, ub = index_range
            i = ops.index_expr(i, torch.int32)
            lb = ops.index_expr(lb, torch.int64)
            ub = ops.index_expr(ub, torch.int64)
            return ops.and_(ops.ge(i, lb), ops.le(i, ub))
        center_x, center_y = (x + top, y + left)
        top_reflect_x, left_reflect_y = (top - x, left - y)
        bot_reflect_x, right_reflect_y = (2 * h + top - x, 2 * w + left - y)
        range_cx = (center_x, 0, h + top + bot)
        range_cy = (center_y, 0, w + left + right)
        cond = ops.and_(index_range_condition(range_cx), index_range_condition(range_cy))
        grad = ops.masked(cond, lambda: load_from_output(center_x, center_y), 0.0)

        def accumulate(out_x, out_y, index_range1, index_range2=None):
            nonlocal grad
            upper_less_than_lower1 = index_range1[2] < index_range1[1]
            if isinstance(upper_less_than_lower1, bool) and upper_less_than_lower1:
                return
            cond = index_range_condition(index_range1)
            if index_range2 is not None:
                upper_less_than_lower2 = index_range2[2] < index_range2[1]
                if isinstance(upper_less_than_lower2, bool) and upper_less_than_lower2:
                    return
                cond = ops.and_(cond, index_range_condition(index_range2))
            g = ops.masked(cond, lambda: load_from_output(out_x, out_y), 0.0)
            grad = ops.add(grad, g)
        accumulate(center_x, left_reflect_y, range_cx, (y, 1, left))
        accumulate(center_x, right_reflect_y, range_cx, (y, w - right, w - 1))
        accumulate(top_reflect_x, center_y, (x, 1, top), range_cy)
        accumulate(bot_reflect_x, center_y, (x, h - bot, h - 1), range_cy)
        accumulate(top_reflect_x, left_reflect_y, (x, 1, top), (y, 1, left))
        accumulate(top_reflect_x, right_reflect_y, (x, 1, top), (y, w - right, w - 1))
        accumulate(bot_reflect_x, left_reflect_y, (x, h - bot, h - 1), (y, 1, left))
        accumulate(bot_reflect_x, right_reflect_y, (x, h - bot, h - 1), (y, w - right, w - 1))
        return grad
    return Pointwise.create(device=grad_output.get_device(), dtype=grad_output.get_dtype(), inner_fn=fn, ranges=list(x.get_size()))