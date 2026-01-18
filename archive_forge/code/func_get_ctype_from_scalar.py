import ast
from typing import Any, Callable, Mapping, Optional, Tuple, Type
import numpy
import numpy.typing as npt
import operator
import cupy
from cupy._logic import ops
from cupy._math import arithmetic
from cupy._logic import comparison
from cupy._binary import elementwise
from cupy import _core
from cupyx.jit import _cuda_types
def get_ctype_from_scalar(mode: str, x: Any) -> _cuda_types.Scalar:
    if isinstance(x, numpy.generic):
        return _cuda_types.Scalar(x.dtype)
    if mode == 'numpy':
        if isinstance(x, bool):
            return _cuda_types.Scalar(numpy.bool_)
        if isinstance(x, int):
            return _cuda_types.Scalar(int)
        if isinstance(x, float):
            return _cuda_types.Scalar(numpy.float64)
        if isinstance(x, complex):
            return _cuda_types.Scalar(numpy.complex128)
    if mode == 'cuda':
        if isinstance(x, bool):
            return _cuda_types.Scalar(numpy.bool_)
        if isinstance(x, int):
            if -(1 << 31) <= x < 1 << 31:
                return _cuda_types.Scalar(numpy.int32)
            return _cuda_types.Scalar(numpy.int64)
        if isinstance(x, float):
            return _cuda_types.Scalar(numpy.float32)
        if isinstance(x, complex):
            return _cuda_types.Scalar(numpy.complex64)
    raise NotImplementedError(f'{x} is not scalar object.')