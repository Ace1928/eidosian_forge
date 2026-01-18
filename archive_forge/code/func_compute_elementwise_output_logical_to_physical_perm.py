from __future__ import annotations
import operator
import warnings
import weakref
from contextlib import nullcontext
from enum import Enum
from functools import cmp_to_key, reduce
from typing import (
import torch
from torch import sym_float, sym_int, sym_max
def compute_elementwise_output_logical_to_physical_perm(*tensors, _skip_checks=False) -> List[int]:
    if not _skip_checks and len(tensors) == 0:
        msg = "Can't compute elementwise output strides for zero tensors!"
        raise ValueError(msg)
    if not _skip_checks:
        check_same_shape(*tensors, allow_cpu_scalar_tensors=True)
    if not _skip_checks:
        tensors = tuple((a for a in tensors if isinstance(a, TensorLike) and (not is_cpu_scalar_tensor(a))))
    if len(tensors) == 0:
        return []
    ndim = tensors[0].ndim
    if ndim == 0:
        return []
    if ndim == 1:
        return [0]
    is_contiguous = True
    for t in tensors:
        is_contiguous = is_contiguous and t.is_contiguous(memory_format=torch.contiguous_format)
    if is_contiguous:
        return list(range(ndim))
    shape = tensors[0].shape

    def should_swap(idx_a, idx_b):
        for tensor in tensors:
            stride_a = tensor.stride()[idx_a]
            stride_b = tensor.stride()[idx_b]
            if stride_a == 0 or stride_b == 0:
                continue
            if stride_a < stride_b:
                return -1
            if stride_a > stride_b:
                return 1
            if shape[idx_a] > shape[idx_b]:
                return 1
        return 0
    perm = list(reversed(range(ndim)))
    for i in range(1, ndim):
        dim1 = i
        for dim0 in reversed(range(i)):
            comparison = should_swap(perm[dim0], perm[dim1])
            if comparison > 0:
                perm[dim0], perm[dim1] = (perm[dim1], perm[dim0])
                dim1 = dim0
            elif comparison < 0:
                break
    return list(reversed(perm))