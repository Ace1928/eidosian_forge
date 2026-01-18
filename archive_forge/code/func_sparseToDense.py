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
def sparseToDense(x, out=None):
    """Converts sparse matrix to a dense matrix.

    Args:
        x (cupyx.scipy.sparse.spmatrix): A sparse matrix to convert.
        out (cupy.ndarray or None): A dense metrix to store the result.
            It must be F-contiguous.

    Returns:
        cupy.ndarray: A converted dense matrix.

    """
    if not check_availability('sparseToDense'):
        raise RuntimeError('sparseToDense is not available.')
    dtype = x.dtype
    assert dtype.char in 'fdFD'
    if out is None:
        out = _cupy.zeros(x.shape, dtype=dtype, order='F')
    else:
        assert out.flags.f_contiguous
        assert out.dtype == dtype
    desc_x = SpMatDescriptor.create(x)
    desc_out = DnMatDescriptor.create(out)
    algo = _cusparse.CUSPARSE_SPARSETODENSE_ALG_DEFAULT
    handle = _device.get_cusparse_handle()
    buff_size = _cusparse.sparseToDense_bufferSize(handle, desc_x.desc, desc_out.desc, algo)
    buff = _cupy.empty(buff_size, _cupy.int8)
    if _runtime.is_hip:
        if x.nnz == 0:
            raise ValueError('hipSPARSE currently cannot handle sparse matrices with null ptrs')
    _cusparse.sparseToDense(handle, desc_x.desc, desc_out.desc, algo, buff.data.ptr)
    return out