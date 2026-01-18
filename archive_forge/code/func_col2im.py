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
@register_decomposition(aten.col2im)
@out_wrapper()
@pw_cast_for_opmath
def col2im(input: Tensor, output_size: List[int], kernel_size: List[int], dilation: List[int], padding: List[int], stride: List[int]) -> Tensor:
    torch._check(len(output_size) == 2, lambda: 'only 2D output_size supported')
    torch._check(len(kernel_size) == 2, lambda: 'only 2D kernel supported')
    torch._check(len(dilation) == 2, lambda: 'only 2D dilation supported')
    torch._check(len(padding) == 2, lambda: 'only 2D padding supported')
    torch._check(len(stride) == 2, lambda: 'only 2D stride supported')

    def check_positive(param, param_name, strict=True):
        cond = all((p > 0 for p in param)) if strict else all((p >= 0 for p in param))
        torch._check(cond, lambda: '{param_name} should be greater than zero, but got {param}')
    check_positive(kernel_size, 'kernel_size')
    check_positive(dilation, 'dilation')
    check_positive(padding, 'padding', strict=False)
    check_positive(stride, 'stride')
    check_positive(output_size, 'output_size')
    shape = input.shape
    ndim = len(shape)
    torch._check(ndim in (2, 3) and all((d != 0 for d in shape[-2:])), lambda: f'Expected 2D or 3D (batch mode) tensor for input with possible 0 batch size and non-zero dimensions, but got: {tuple(shape)}')
    prod_kernel_size = kernel_size[0] * kernel_size[1]
    torch._check(shape[-2] % prod_kernel_size == 0, lambda: f"Expected size of input's first non-batch dimension to be divisible by the product of kernel_size, but got input.shape[-2] = {shape[-2]} and kernel_size={kernel_size}")
    col = [1 + (out + 2 * pad - dil * (ker - 1) - 1) // st for out, pad, dil, ker, st in zip(output_size, padding, dilation, kernel_size, stride)]
    L = col[0] * col[1]
    torch._check(shape[-1] == L, lambda: f'Given output_size={output_size}, kernel_size={kernel_size}, dilation={dilation}, padding={padding}, stride={stride}, expected input.size(-1) to be {L} but got {shape[-1]}.')
    torch._check(L > 0, lambda: f'Given output_size={output_size}, kernel_size={kernel_size}, dilation={dilation}, padding={padding}, stride={stride}, expected input.size(-1) to be {L} but got {shape[-1]}.')
    batched_input = ndim == 3
    if not batched_input:
        input = input.unsqueeze(0)
    shape = input.shape
    out_h, out_w = output_size
    stride_h, stride_w = stride
    padding_h, padding_w = padding
    dilation_h, dilation_w = dilation
    kernel_h, kernel_w = kernel_size
    input = input.reshape([shape[0], shape[1] // prod_kernel_size] + kernel_size + col)
    input = input.permute(0, 1, 2, 4, 3, 5)
    indices_row = _im2col_col2im_indices_along_dim(out_h, kernel_h, dilation_h, padding_h, stride_h, input.device)
    indices_row = _unsqueeze_to_dim(indices_row, 4)
    indices_col = _im2col_col2im_indices_along_dim(out_w, kernel_w, dilation_w, padding_w, stride_w, input.device)
    output_padded_size = [o + 2 * p for o, p in zip(output_size, padding)]
    output = input.new_zeros([shape[0], shape[1] // prod(kernel_size)] + output_padded_size)
    idx = (None, None, indices_row, indices_col)
    output = aten._unsafe_index_put(output, idx, input, accumulate=True)
    output = F.pad(output, (-padding_w, -padding_w, -padding_h, -padding_h))
    if not batched_input:
        output = output.squeeze(0)
    return output