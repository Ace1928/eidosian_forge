from array import array as native_array
import ctypes
import warnings
import operator
from functools import reduce # pylint: disable=redefined-builtin
import numpy as np
from ..base import _LIB, numeric_types, integer_types
from ..base import c_str, c_array, c_array_buf, c_handle_array, mx_real_t
from ..base import mx_uint, NDArrayHandle, check_call, DLPackHandle, mx_int, mx_int64
from ..base import ctypes2buffer
from ..runtime import Features
from ..context import Context, current_context
from ..util import is_np_array
from . import _internal
from . import op
from ._internal import NDArrayBase
def lesser_equal(lhs, rhs):
    """Returns the result of element-wise **lesser than or equal to** (<=) comparison
    operation with broadcasting.

    For each element in input arrays, return 1(true) if lhs elements are
    lesser than equal to rhs, otherwise return 0(false).

    Equivalent to ``lhs <= rhs`` and ``mx.nd.broadcast_lesser_equal(lhs, rhs)``.

    .. note::

       If the corresponding dimensions of two arrays have the same size or one of them has size 1,
       then the arrays are broadcastable to a common shape.

    Parameters
    ----------
    lhs : scalar or mxnet.ndarray.array
        First array to be compared.
    rhs : scalar or mxnet.ndarray.array
         Second array to be compared. If ``lhs.shape != rhs.shape``, they must be
        broadcastable to a common shape.

    Returns
    -------
    NDArray
        Output array of boolean values.

    Examples
    --------
    >>> x = mx.nd.ones((2,3))
    >>> y = mx.nd.arange(2).reshape((2,1))
    >>> z = mx.nd.arange(2).reshape((1,2))
    >>> x.asnumpy()
    array([[ 1.,  1.,  1.],
           [ 1.,  1.,  1.]], dtype=float32)
    >>> y.asnumpy()
    array([[ 0.],
           [ 1.]], dtype=float32)
    >>> z.asnumpy()
    array([[ 0.,  1.]], dtype=float32)
    >>> (x <= 1).asnumpy()
    array([[ 1.,  1.,  1.],
           [ 1.,  1.,  1.]], dtype=float32)
    >>> (x <= y).asnumpy()
    array([[ 0.,  0.,  0.],
           [ 1.,  1.,  1.]], dtype=float32)
    >>> mx.nd.lesser_equal(x, y).asnumpy()
    array([[ 0.,  0.,  0.],
           [ 1.,  1.,  1.]], dtype=float32)
    >>> (z <= y).asnumpy()
    array([[ 1.,  0.],
           [ 1.,  1.]], dtype=float32)
    """
    return _ufunc_helper(lhs, rhs, op.broadcast_lesser_equal, lambda x, y: 1 if x <= y else 0, _internal._lesser_equal_scalar, _internal._greater_equal_scalar)