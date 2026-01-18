from types import BuiltinFunctionType
import ctypes
from functools import partial
import numpy as np
from numba.core import types
from numba.core.errors import TypingError
from numba.core.typing import templates
from numba.np import numpy_support
def map_type(cffi_type, use_record_dtype=False):
    """
    Map CFFI type to numba type.

    Parameters
    ----------
    cffi_type:
        The CFFI type to be converted.
    use_record_dtype: bool (default: False)
        When True, struct types are mapped to a NumPy Record dtype.

    """
    primed_map_type = partial(map_type, use_record_dtype=use_record_dtype)
    kind = getattr(cffi_type, 'kind', '')
    if kind == 'union':
        raise TypeError('No support for CFFI union')
    elif kind == 'function':
        if cffi_type.ellipsis:
            raise TypeError('vararg function is not supported')
        restype = primed_map_type(cffi_type.result)
        argtypes = [primed_map_type(arg) for arg in cffi_type.args]
        return templates.signature(restype, *argtypes)
    elif kind == 'pointer':
        pointee = cffi_type.item
        if pointee.kind == 'void':
            return types.voidptr
        else:
            return types.CPointer(primed_map_type(pointee))
    elif kind == 'array':
        dtype = primed_map_type(cffi_type.item)
        nelem = cffi_type.length
        return types.NestedArray(dtype=dtype, shape=(nelem,))
    elif kind == 'struct' and use_record_dtype:
        return map_struct_to_record_dtype(cffi_type)
    else:
        result = _type_map().get(cffi_type)
        if result is None:
            raise TypeError(cffi_type)
        return result