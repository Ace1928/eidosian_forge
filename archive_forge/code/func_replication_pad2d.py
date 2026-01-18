import functools
import numbers
import operator
import sys
from enum import Enum
from functools import partial, reduce
from itertools import chain, product
from typing import Callable, cast, Iterable, List, Optional, Tuple, Union
import torch
import torch._prims as prims
import torch._prims_common as utils
import torch.nn.functional as F
from torch import sym_float, sym_int, Tensor
from torch._decomp import register_decomposition
from torch._higher_order_ops.out_dtype import out_dtype
from torch._prims_common import IntLike, NumberType, TensorLike, TensorSequenceType
from torch._prims_common.wrappers import (
from torch.utils import _pytree as pytree
from torch.utils._pytree import tree_map
@register_decomposition(aten.replication_pad2d.default)
@pw_cast_for_opmath
def replication_pad2d(input: Tensor, padding: List[int]) -> Tensor:
    pad_left = padding[0]
    pad_right = padding[1]
    pad_top = padding[2]
    pad_bottom = padding[3]
    input_mid = input
    input_mid_tb = input
    input_mid_lr = input
    if pad_left < 0:
        input_mid = input_mid[..., -pad_left:]
        input_mid_tb = input_mid_tb[..., -pad_left:]
        pad_left = 0
    if pad_right < 0:
        input_mid = input_mid[..., :pad_right]
        input_mid_tb = input_mid_tb[..., :pad_right]
        pad_right = 0
    if pad_top < 0:
        input_mid = input_mid[..., -pad_top:, :]
        input_mid_lr = input_mid_lr[..., -pad_top:, :]
        pad_top = 0
    if pad_bottom < 0:
        input_mid = input_mid[..., :pad_bottom, :]
        input_mid_lr = input_mid_lr[..., :pad_bottom, :]
        pad_bottom = 0
    batch_dims_no_repeat = (1,) * (input.dim() - 2)
    repeat_top_left = batch_dims_no_repeat + (pad_top, pad_left)
    repeat_top_middle = batch_dims_no_repeat + (pad_top, 1)
    repeat_top_right = batch_dims_no_repeat + (pad_top, pad_right)
    top_rows = torch.cat([input[..., [0], :][..., [0]].repeat(repeat_top_left), input_mid_tb[..., [0], :].repeat(repeat_top_middle), input[..., [0], :][..., [-1]].repeat(repeat_top_right)], dim=-1)
    repeat_middle_left = batch_dims_no_repeat + (1, pad_left)
    repeat_middle_right = batch_dims_no_repeat + (1, pad_right)
    middle_rows = torch.cat([input_mid_lr[..., [0]].repeat(repeat_middle_left), input_mid, input_mid_lr[..., [-1]].repeat(repeat_middle_right)], dim=-1)
    repeat_bottom_left = batch_dims_no_repeat + (pad_bottom, pad_left)
    repeat_bottom_middle = batch_dims_no_repeat + (pad_bottom, 1)
    repeat_bottom_right = batch_dims_no_repeat + (pad_bottom, pad_right)
    bottom_rows = torch.cat([input[..., [-1], :][..., [0]].repeat(repeat_bottom_left), input_mid_tb[..., [-1], :].repeat(repeat_bottom_middle), input[..., [-1], :][..., [-1]].repeat(repeat_bottom_right)], dim=-1)
    return torch.cat([top_rows, middle_rows, bottom_rows], dim=-2)