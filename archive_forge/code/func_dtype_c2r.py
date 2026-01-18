from __future__ import annotations
import scipy.ndimage
import scipy.sparse
import numpy as np
import numba
from numpy.lib.stride_tricks import as_strided
from .._cache import cache
from .exceptions import ParameterError
from .deprecation import Deprecated
from numpy.typing import ArrayLike, DTypeLike
from typing import (
from typing_extensions import Literal
from .._typing import _SequenceLike, _FloatLike_co, _ComplexLike_co
def dtype_c2r(d: DTypeLike, *, default: Optional[type]=np.float32) -> DTypeLike:
    """Find the real numpy dtype corresponding to a complex dtype.

    This is used to maintain numerical precision and memory footprint
    when constructing real arrays from complex-valued data
    (e.g. in an inverse Fourier transform).

    A `complex64` (single-precision) type maps to `float32`,
    while a `complex128` (double-precision) maps to `float64`.

    Parameters
    ----------
    d : np.dtype
        The complex-valued dtype to convert to real.
        If ``d`` is a real (float) type already, it will be returned.
    default : np.dtype, optional
        The default real target type, if ``d`` does not match a
        known dtype

    Returns
    -------
    d_r : np.dtype
        The real dtype

    See Also
    --------
    dtype_r2c
    numpy.dtype

    Examples
    --------
    >>> librosa.util.dtype_r2c(np.complex64)
    dtype('float32')

    >>> librosa.util.dtype_r2c(np.float32)
    dtype('float32')

    >>> librosa.util.dtype_r2c(np.int16)
    dtype('float32')

    >>> librosa.util.dtype_r2c(np.complex128)
    dtype('float64')
    """
    mapping: Dict[DTypeLike, type] = {np.dtype(np.complex64): np.float32, np.dtype(np.complex128): np.float64, np.dtype(complex): np.dtype(float).type}
    dt = np.dtype(d)
    if dt.kind == 'f':
        return dt
    return np.dtype(mapping.get(dt, default))