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
def csrmvExIsAligned(a, x, y=None):
    """Check if the pointers of arguments for csrmvEx are aligned or not

    Args:
        a (cupyx.cusparse.csr_matrix): Matrix A.
        x (cupy.ndarray): Vector x.
        y (cupy.ndarray or None): Vector y.

        Check if a, x, y pointers are aligned by 128 bytes as
        required by csrmvEx.

    Returns:
        bool:
        ``True`` if all pointers are aligned.
        ``False`` if otherwise.

    """
    if a.data.data.ptr % 128 != 0:
        return False
    if a.indptr.data.ptr % 128 != 0:
        return False
    if a.indices.data.ptr % 128 != 0:
        return False
    if x.data.ptr % 128 != 0:
        return False
    if y is not None and y.data.ptr % 128 != 0:
        return False
    return True