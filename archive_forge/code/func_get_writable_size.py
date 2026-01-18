import ctypes
import sys
from typing import Any, Callable, Dict, Optional, Tuple, TYPE_CHECKING, TypeVar, Union
import weakref
from pyglet.media.drivers.pulse import lib_pulseaudio as pa
from pyglet.media.exceptions import MediaException
from pyglet.util import debug_print
def get_writable_size(self) -> int:
    assert self._pa_stream is not None
    r = pa.pa_stream_writable_size(self._pa_stream)
    if r == PA_INVALID_WRITABLE_SIZE:
        self.context().raise_error()
    return r