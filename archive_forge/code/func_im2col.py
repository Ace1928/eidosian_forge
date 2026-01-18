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
@register_decomposition(aten.im2col)
@out_wrapper()
@pw_cast_for_opmath
def im2col(input: Tensor, kernel_size: List[int], dilation: List[int], padding: List[int], stride: List[int]) -> Tensor:
    torch._check(len(kernel_size) == 2, lambda: 'im2col(): only 2D kernel supported')
    torch._check(len(dilation) == 2, lambda: 'im2col(): only 2D dilation supported')
    torch._check(len(padding) == 2, lambda: 'im2col(): only 2D padding supported')
    torch._check(len(stride) == 2, lambda: 'im2col(): only 2D stride supported')

    def check_positive(param, param_name, strict=True):
        cond = all((p > 0 for p in param)) if strict else all((p >= 0 for p in param))
        torch._check(cond, lambda: "{param_name} should be greater {'than' zero, but got {param}")
    check_positive(kernel_size, 'kernel_size')
    check_positive(dilation, 'dilation')
    check_positive(dilation, 'padding', strict=False)
    check_positive(stride, 'stride')
    shape = input.shape
    ndim = len(shape)
    torch._check(ndim in (3, 4) and all((d != 0 for d in shape[-3:])), lambda: f'Expected 3D or 4D (batch mode) tensor for input with possible 0 batch size and non-zero dimensions, but got: {tuple(shape)}')
    output_size = tuple((1 + (out + 2 * pad - dil * (ker - 1) - 1) // st for out, pad, dil, ker, st in zip(shape[-2:], padding, dilation, kernel_size, stride)))
    torch._check(all((c > 0 for c in output_size)), lambda: f'Given an input with spacial size {tuple(shape[-2:])}, kernel_size={kernel_size}, dilation={dilation}, padding={padding}, stride={stride}, the calculated shape of the array of sliding blocks is {output_size}, but its components must be at least one.')
    batched_input = ndim == 4
    if not batched_input:
        input = input.unsqueeze(0)
    batch_dim, channel_dim, input_h, input_w = input.shape
    stride_h, stride_w = stride
    padding_h, padding_w = padding
    dilation_h, dilation_w = dilation
    kernel_h, kernel_w = kernel_size
    blocks_row_indices = _im2col_col2im_indices_along_dim(input_h, kernel_h, dilation_h, padding_h, stride_h, input.device)
    blocks_col_indices = _im2col_col2im_indices_along_dim(input_w, kernel_w, dilation_w, padding_w, stride_w, input.device)
    padded_input = F.pad(input, (padding_w, padding_w, padding_h, padding_h))
    blocks_row_indices = blocks_row_indices.unsqueeze(-1).unsqueeze(-1)
    output = padded_input[:, :, blocks_row_indices, blocks_col_indices]
    output = output.permute(0, 1, 2, 4, 3, 5)
    num_blocks_row = blocks_row_indices.size(1)
    num_blocks_col = blocks_col_indices.size(1)
    output = output.reshape(batch_dim, channel_dim * kernel_h * kernel_w, num_blocks_row * num_blocks_col)
    if not batched_input:
        output = output.squeeze(0)
    return output