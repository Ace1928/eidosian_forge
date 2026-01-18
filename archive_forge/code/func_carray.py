import collections
import ctypes
import re
import numpy as np
from numba.core import errors, types
from numba.core.typing.templates import signature
from numba.np import npdatetime_helpers
from numba.core.errors import TypingError
from numba.core.cgutils import is_nonelike   # noqa: F401
def carray(ptr, shape, dtype=None):
    """
    Return a Numpy array view over the data pointed to by *ptr* with the
    given *shape*, in C order.  If *dtype* is given, it is used as the
    array's dtype, otherwise the array's dtype is inferred from *ptr*'s type.
    """
    from numba.core.typing.ctypes_utils import from_ctypes
    try:
        ptr = ptr._as_parameter_
    except AttributeError:
        pass
    if dtype is not None:
        dtype = np.dtype(dtype)
    if isinstance(ptr, ctypes.c_void_p):
        if dtype is None:
            raise TypeError('explicit dtype required for void* argument')
        p = ptr
    elif isinstance(ptr, ctypes._Pointer):
        ptrty = from_ctypes(ptr.__class__)
        assert isinstance(ptrty, types.CPointer)
        ptr_dtype = as_dtype(ptrty.dtype)
        if dtype is not None and dtype != ptr_dtype:
            raise TypeError("mismatching dtype '%s' for pointer %s" % (dtype, ptr))
        dtype = ptr_dtype
        p = ctypes.cast(ptr, ctypes.c_void_p)
    else:
        raise TypeError('expected a ctypes pointer, got %r' % (ptr,))
    nbytes = dtype.itemsize * np.product(shape, dtype=np.intp)
    return _get_array_from_ptr(p, nbytes, dtype).reshape(shape)