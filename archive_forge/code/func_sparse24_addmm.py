import contextlib
import ctypes
import glob
import warnings
from functools import partial
from pathlib import Path
from typing import Any, Callable, Optional, Tuple, TypeVar, cast
import torch
from .common import BaseOperator, get_operator, get_xformers_operator, register_operator
def sparse24_addmm(func, types, args=(), kwargs=None) -> torch.Tensor:
    assert len(args) == 3
    bias, A, B = args
    if A.ndim != 2 or B.ndim != 2:
        raise NotImplementedError('`Sparse24Tensor` matmul: Broadcasting is not implemented')
    if bias.ndim != 1:
        raise NotImplementedError(f'`Sparse24Tensor` matmul: only bias dim=1 supported. Shape={bias.shape}')
    if isinstance(A, Sparse24Tensor):
        raise NotImplementedError('`Sparse24Tensor` matmul: only operand B of `addmm` can be sparse')
    B_t = B.t()
    assert isinstance(B_t, Sparse24Tensor)
    return B_t._mm(A.t(), bias=bias, prefer_col_major_output=True).t()