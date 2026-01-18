import platform
from ctypes import (POINTER, c_char_p, c_bool, c_void_p,
from llvmlite.binding import ffi, targets, object_file
def set_object_cache(self, notify_func=None, getbuffer_func=None):
    """
        Set the object cache "notifyObjectCompiled" and "getBuffer"
        callbacks to the given Python functions.
        """
    self._object_cache_notify = notify_func
    self._object_cache_getbuffer = getbuffer_func
    self._object_cache = _ObjectCacheRef(self)
    ffi.lib.LLVMPY_SetObjectCache(self, self._object_cache)