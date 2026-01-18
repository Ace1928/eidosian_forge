import functools as _functools
import numpy as _numpy
import platform as _platform
import cupy as _cupy
from cupy_backends.cuda.api import driver as _driver
from cupy_backends.cuda.api import runtime as _runtime
from cupy_backends.cuda.libs import cusparse as _cusparse
from cupy._core import _dtype
from cupy.cuda import device as _device
from cupy.cuda import stream as _stream
from cupy import _util
import cupyx.scipy.sparse
def csc2dense(x, out=None):
    """Converts CSC-matrix to a dense matrix.

    Args:
        x (cupyx.scipy.sparse.csc_matrix): A sparse matrix to convert.
        out (cupy.ndarray or None): A dense metrix to store the result.
            It must be F-contiguous.

    Returns:
        cupy.ndarray: Converted result.

    """
    if not check_availability('csc2dense'):
        raise RuntimeError('csc2dense is not available.')
    dtype = x.dtype
    assert dtype.char in 'fdFD'
    if out is None:
        out = _cupy.empty(x.shape, dtype=dtype, order='F')
    else:
        assert out.flags.f_contiguous
    handle = _device.get_cusparse_handle()
    _call_cusparse('csc2dense', x.dtype, handle, x.shape[0], x.shape[1], x._descr.descriptor, x.data.data.ptr, x.indices.data.ptr, x.indptr.data.ptr, out.data.ptr, x.shape[0])
    return out