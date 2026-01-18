import ctypes
from ..base import _LIB, check_call, py_str, c_str, string_types, mx_uint, NDArrayHandle
from ..base import c_array, c_handle_array, c_str_array
from .ndarray import NDArray
from .ndarray import array as _array
from .ndarray import empty as _empty_ndarray
from .ndarray import zeros as _zeros_ndarray
from .sparse import zeros as _zeros_sparse_ndarray
from .sparse import empty as _empty_sparse_ndarray
from .sparse import array as _sparse_array
from .sparse import _ndarray_cls
def load_frombuffer(buf):
    """Loads an array dictionary or list from a buffer

    See more details in ``save``.

    Parameters
    ----------
    buf : str
        Buffer containing contents of a file as a string or bytes.

    Returns
    -------
    list of NDArray, RowSparseNDArray or CSRNDArray, or     dict of str to NDArray, RowSparseNDArray or CSRNDArray
        Loaded data.
    """
    if not isinstance(buf, string_types + tuple([bytes])):
        raise TypeError('buf required to be a string or bytes')
    out_size = mx_uint()
    out_name_size = mx_uint()
    handles = ctypes.POINTER(NDArrayHandle)()
    names = ctypes.POINTER(ctypes.c_char_p)()
    check_call(_LIB.MXNDArrayLoadFromBuffer(buf, mx_uint(len(buf)), ctypes.byref(out_size), ctypes.byref(handles), ctypes.byref(out_name_size), ctypes.byref(names)))
    if out_name_size.value == 0:
        return [_ndarray_cls(NDArrayHandle(handles[i])) for i in range(out_size.value)]
    else:
        assert out_name_size.value == out_size.value
        return dict(((py_str(names[i]), _ndarray_cls(NDArrayHandle(handles[i]))) for i in range(out_size.value)))