import ctypes
import weakref
from . import heap
from . import get_context
from .context import reduction, assert_spawning
def rebuild_ctype(type_, wrapper, length):
    if length is not None:
        type_ = type_ * length
    _ForkingPickler.register(type_, reduce_ctype)
    buf = wrapper.create_memoryview()
    obj = type_.from_buffer(buf)
    obj._wrapper = wrapper
    return obj