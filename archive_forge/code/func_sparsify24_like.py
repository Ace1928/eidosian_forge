import contextlib
import ctypes
import glob
import warnings
from functools import partial
from pathlib import Path
from typing import Any, Callable, Optional, Tuple, TypeVar, cast
import torch
from .common import BaseOperator, get_operator, get_xformers_operator, register_operator
@allow_in_graph
def sparsify24_like(x: torch.Tensor, pattern: torch.Tensor, out_dense: bool=False) -> Sparse24Tensor:
    if not isinstance(pattern, Sparse24Tensor):
        raise ValueError(f'`pattern` must be a `Sparse24Tensor` but got a {type(pattern)}')
    if not pattern.threads_masks.is_contiguous():
        return _Sparsify24LikeFunc.apply(x.t(), pattern.t(), out_dense).t()
    return _Sparsify24LikeFunc.apply(x, pattern, out_dense)