import sys, types
from .lock import allocate_lock
from .error import CDefError
from . import model
def def_extern(self, *args, **kwds):
    raise ValueError('ffi.def_extern() is only available on API-mode FFI objects')