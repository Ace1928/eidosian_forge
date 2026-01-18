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
def asscipy(self):
    """Returns a ``scipy.sparse.csr.csr_matrix`` object with value copied from this array

        Examples
        --------
        >>> x = mx.nd.sparse.zeros('csr', (2,3))
        >>> y = x.asscipy()
        >>> type(y)
        <type 'scipy.sparse.csr.csr_matrix'>
        >>> y
        <2x3 sparse matrix of type '<type 'numpy.float32'>'
        with 0 stored elements in Compressed Sparse Row format>
        """
    data = self.data.asnumpy()
    indices = self.indices.asnumpy()
    indptr = self.indptr.asnumpy()
    if not spsp:
        raise ImportError('scipy could not be imported. Please make sure that the scipy is installed.')
    return spsp.csr_matrix((data, indices, indptr), shape=self.shape, dtype=self.dtype)