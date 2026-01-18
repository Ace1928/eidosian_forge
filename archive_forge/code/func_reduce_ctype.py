import ctypes
import weakref
from . import heap
from . import get_context
from .context import reduction, assert_spawning
def reduce_ctype(obj):
    assert_spawning(obj)
    if isinstance(obj, ctypes.Array):
        return (rebuild_ctype, (obj._type_, obj._wrapper, obj._length_))
    else:
        return (rebuild_ctype, (type(obj), obj._wrapper, None))