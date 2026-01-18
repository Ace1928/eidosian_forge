import contextlib
import typing
from . import openrlib
from _cffi_backend import FFI  # type: ignore
def protect(self, cdata: FFI.CData):
    """Pass-through function that adds the R object to the short-term
        stack of objects protected from garbase collection."""
    cdata = openrlib.rlib.Rf_protect(cdata)
    self._counter += 1
    return cdata