import ctypes
import sys
from typing import Any, Callable, Dict, Optional, Tuple, TYPE_CHECKING, TypeVar, Union
import weakref
from pyglet.media.drivers.pulse import lib_pulseaudio as pa
from pyglet.media.exceptions import MediaException
from pyglet.util import debug_print
def begin_write(self, nbytes: Optional[int]=None) -> Tuple[ctypes.c_void_p, int]:
    context = self.context()
    assert context is not None
    addr = ctypes.c_void_p()
    nbytes_st = ctypes.c_size_t(_SIZE_T_MAX if nbytes is None else nbytes)
    context.check(pa.pa_stream_begin_write(self._pa_stream, ctypes.byref(addr), ctypes.byref(nbytes_st)))
    context.check_ptr_not_null(addr)
    assert _debug(f'PulseAudioStream: begin_write nbytes={nbytes} nbytes_n={nbytes_st.value}')
    return (addr, nbytes_st.value)