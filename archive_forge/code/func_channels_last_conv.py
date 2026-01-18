from __future__ import annotations
import functools
import logging
from typing import cast, List, Optional, Sequence, Tuple, TypedDict
import torch
from .. import config, ir
from ..ir import TensorBox
from ..lowering import (
from ..select_algorithm import (
from ..utils import (
from ..virtualized import V
from .mm_common import filtered_configs
def channels_last_conv():
    if V.graph.layout_opt and ndim == 2:
        return True
    layout = conv_layout(x, weight, None, **kwargs)
    req_stride_order = ir.get_stride_order(V.graph.sizevars.size_hints(layout.stride))
    return req_stride_order == ir.NHWC_STRIDE_ORDER