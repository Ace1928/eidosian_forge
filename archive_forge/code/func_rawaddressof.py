import ctypes, ctypes.util, operator, sys
from . import model
def rawaddressof(self, BTypePtr, cdata, offset=None):
    if isinstance(cdata, CTypesBaseStructOrUnion):
        ptr = ctypes.pointer(type(cdata)._to_ctypes(cdata))
    elif isinstance(cdata, CTypesGenericPtr):
        if offset is None or not issubclass(type(cdata)._BItem, CTypesBaseStructOrUnion):
            raise TypeError('unexpected cdata type')
        ptr = type(cdata)._to_ctypes(cdata)
    elif isinstance(cdata, CTypesGenericArray):
        ptr = type(cdata)._to_ctypes(cdata)
    else:
        raise TypeError("expected a <cdata 'struct-or-union'>")
    if offset:
        ptr = ctypes.cast(ctypes.c_void_p(ctypes.cast(ptr, ctypes.c_void_p).value + offset), type(ptr))
    return BTypePtr._from_ctypes(ptr)