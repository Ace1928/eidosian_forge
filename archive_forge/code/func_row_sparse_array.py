import ctypes
import warnings
import operator
from array import array as native_array
import numpy as np
from ..base import NotSupportedForSparseNDArray
from ..base import _LIB, numeric_types
from ..base import c_array_buf, mx_real_t, integer_types
from ..base import NDArrayHandle, check_call
from ..context import Context, current_context
from . import _internal
from . import op
from ._internal import _set_ndarray_class
from .ndarray import NDArray, _storage_type, _DTYPE_NP_TO_MX, _DTYPE_MX_TO_NP
from .ndarray import _STORAGE_TYPE_STR_TO_ID, _STORAGE_TYPE_ROW_SPARSE, _STORAGE_TYPE_CSR, _int64_enabled
from .ndarray import _STORAGE_TYPE_UNDEFINED, _STORAGE_TYPE_DEFAULT
from .ndarray import zeros as _zeros_ndarray
from .ndarray import array as _array
from .ndarray import _ufunc_helper
def row_sparse_array(arg1, shape=None, ctx=None, dtype=None):
    """Creates a `RowSparseNDArray`, a multidimensional row sparse array with a set of     tensor slices at given indices.

    The RowSparseNDArray can be instantiated in several ways:

    - row_sparse_array(D):
        to construct a RowSparseNDArray with a dense ndarray ``D``
        -  **D** (*array_like*) - An object exposing the array interface, an object whose         `__array__` method returns an array, or any (nested) sequence.
        - **ctx** (*Context, optional*) - Device context         (default is the current default context).
        - **dtype** (*str or numpy.dtype, optional*) - The data type of the output array.         The default dtype is ``D.dtype`` if ``D`` is an NDArray or numpy.ndarray,         float32 otherwise.

    - row_sparse_array(S)
        to construct a RowSparseNDArray with a sparse ndarray ``S``
        -  **S** (*RowSparseNDArray*) - A sparse ndarray.
        - **ctx** (*Context, optional*) - Device context         (default is the current default context).
        - **dtype** (*str or numpy.dtype, optional*) - The data type of the output array.         The default dtype is ``S.dtype``.

    - row_sparse_array((D0, D1 .. Dn))
        to construct an empty RowSparseNDArray with shape ``(D0, D1, ... Dn)``
        -  **D0, D1 .. Dn** (*int*) - The shape of the ndarray
        - **ctx** (*Context, optional*) - Device context         (default is the current default context).
        - **dtype** (*str or numpy.dtype, optional*) - The data type of the output array.             The default dtype is float32.

    - row_sparse_array((data, indices))
        to construct a RowSparseNDArray based on the definition of row sparse format         using two separate arrays,         where the `indices` stores the indices of the row slices with non-zeros,
        while the values are stored in `data`. The corresponding NDArray ``dense``
        represented by RowSparseNDArray ``rsp`` has         ``dense[rsp.indices[i], :, :, :, ...] = rsp.data[i, :, :, :, ...]``
        The row indices for are expected to be **sorted in ascending order.**         - **data** (*array_like*) - An object exposing the array interface, which         holds all the non-zero row slices of the array.
        - **indices** (*array_like*) - An object exposing the array interface, which         stores the row index for each row slice with non-zero elements.
        - **shape** (*tuple of int, optional*) - The shape of the array. The default         shape is inferred from the indices and indptr arrays.
        - **ctx** (*Context, optional*) - Device context         (default is the current default context).
        - **dtype** (*str or numpy.dtype, optional*) - The data type of the output array.         The default dtype is float32.

    Parameters
    ----------
    arg1 : NDArray, numpy.ndarray, RowSparseNDArray, tuple of int or tuple of array_like
        The argument to help instantiate the row sparse ndarray. See above for further details.
    shape : tuple of int, optional
        The shape of the row sparse ndarray. (Default value = None)
    ctx : Context, optional
        Device context (default is the current default context).
    dtype : str or numpy.dtype, optional
        The data type of the output array. (Default value = None)

    Returns
    -------
    RowSparseNDArray
        An `RowSparseNDArray` with the `row_sparse` storage representation.

    Examples
    --------
    >>> a = mx.nd.sparse.row_sparse_array(([[1, 2], [3, 4]], [1, 4]), shape=(6, 2))
    >>> a.asnumpy()
    array([[ 0.,  0.],
           [ 1.,  2.],
           [ 0.,  0.],
           [ 0.,  0.],
           [ 3.,  4.],
           [ 0.,  0.]], dtype=float32)

    See Also
    --------
    RowSparseNDArray : MXNet NDArray in row sparse format.
    """
    if isinstance(arg1, tuple):
        arg_len = len(arg1)
        if arg_len < 2:
            raise ValueError('Unexpected length of input tuple: ' + str(arg_len))
        if arg_len > 2:
            _check_shape(arg1, shape)
            return empty('row_sparse', arg1, ctx=ctx, dtype=dtype)
        elif isinstance(arg1[0], integer_types) and isinstance(arg1[1], integer_types):
            _check_shape(arg1, shape)
            return empty('row_sparse', arg1, ctx=ctx, dtype=dtype)
        else:
            return _row_sparse_ndarray_from_definition(arg1[0], arg1[1], shape=shape, ctx=ctx, dtype=dtype)
    elif isinstance(arg1, RowSparseNDArray):
        _check_shape(arg1.shape, shape)
        return array(arg1, ctx=ctx, dtype=dtype)
    elif isinstance(arg1, CSRNDArray):
        raise ValueError('Unexpected input type: CSRNDArray')
    else:
        dtype = _prepare_default_dtype(arg1, dtype)
        dns = _array(arg1, dtype=dtype)
        if ctx is not None and dns.context != ctx:
            dns = dns.as_in_context(ctx)
        _check_shape(dns.shape, shape)
        return dns.tostype('row_sparse')