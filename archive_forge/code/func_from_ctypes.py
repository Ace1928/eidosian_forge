import ctypes
import sys
from numba.core import types
from numba.core.typing import templates
from .typeof import typeof_impl
def from_ctypes(ctypeobj):
    """
    Convert the given ctypes type to a Numba type.
    """
    if ctypeobj is None:
        return types.none
    assert isinstance(ctypeobj, type), ctypeobj

    def _convert_internal(ctypeobj):
        if issubclass(ctypeobj, ctypes._Pointer):
            valuety = _convert_internal(ctypeobj._type_)
            if valuety is not None:
                return types.CPointer(valuety)
        else:
            return _FROM_CTYPES.get(ctypeobj)
    ty = _convert_internal(ctypeobj)
    if ty is None:
        raise TypeError('Unsupported ctypes type: %s' % ctypeobj)
    return ty