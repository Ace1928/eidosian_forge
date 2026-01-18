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
def conv_layout(x: TensorBox, weight: TensorBox, bias: Optional[TensorBox], stride: Sequence[int], padding: tuple[int, ...], dilation: tuple[int, ...], transposed: bool, output_padding: tuple[int, ...], groups: int) -> ir.Layout:
    """Determine output layout for a convolution"""
    with V.graph.fake_mode:
        output = torch.ops.aten.convolution(ir.ir_node_to_tensor(x, guard_shape=True), ir.ir_node_to_tensor(weight, guard_shape=True), ir.ir_node_to_tensor(bias, guard_shape=True), stride, tuple((V.graph.sizevars.size_hint(p) for p in padding)), dilation, transposed, tuple((V.graph.sizevars.size_hint(p) for p in output_padding)), groups)
        sizes = ir.convert_shape_to_inductor(output.size())
        stride = ir.convert_shape_to_inductor(output.stride())
    return ir.FixedLayout(x.get_device(), x.get_dtype(), sizes, stride)