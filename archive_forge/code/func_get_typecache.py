import types
import weakref
from .lock import allocate_lock
from .error import CDefError, VerificationError, VerificationMissing
def get_typecache(backend):
    if isinstance(backend, types.ModuleType):
        return _typecache_cffi_backend
    with global_lock:
        if not hasattr(type(backend), '__typecache'):
            type(backend).__typecache = weakref.WeakValueDictionary()
        return type(backend).__typecache