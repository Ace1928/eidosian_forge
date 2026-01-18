from dataclasses import replace
from enum import Enum
from functools import partial
from typing import Any, List, Mapping, Optional, Set, Tuple, Union
import torch
from ..common import get_xformers_operator, register_operator
from . import attn_bias
from .attn_bias import (
from .common import (
@classmethod
def not_supported_reasons(cls, d: Inputs) -> List[str]:
    reasons = super(BwOp, cls).not_supported_reasons(d)
    matmul_alignment_mn = _minimum_gemm_alignment(d)
    check_lastdim_alignment_stride1(reasons, 'query', d.query, matmul_alignment_mn)
    check_lastdim_alignment_stride1(reasons, 'key', d.key, matmul_alignment_mn)
    check_lastdim_alignment_stride1(reasons, 'value', d.value, matmul_alignment_mn)
    _check_bias_alignment(reasons, d.attn_bias)
    attn_bias_tensor = _get_tensor_bias(d.attn_bias)
    if attn_bias_tensor is not None and attn_bias_tensor.requires_grad:
        if d.query.ndim == 3 and attn_bias_tensor.ndim == 3:
            expected_bias_shape = (*d.query.shape[:2], d.key.shape[1])
        else:
            expected_bias_shape = (d.query.shape[0], d.query.shape[2] if d.query.ndim == 4 else 1, d.query.shape[1], d.key.shape[1])
        if tuple(attn_bias_tensor.shape) != expected_bias_shape:
            reasons.append(f'Broadcasting the `attn_bias` tensor is not supported (shape: {tuple(attn_bias_tensor.shape)}/ expected: {expected_bias_shape})')
    _check_large_shapes(reasons, d)
    reasons.append('Backward is currently not supported by ck-tiled!')
    return reasons