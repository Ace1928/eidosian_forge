import ctypes
import sys
from typing import Any, Callable, Dict, Optional, Tuple, TYPE_CHECKING, TypeVar, Union
import weakref
from pyglet.media.drivers.pulse import lib_pulseaudio as pa
from pyglet.media.exceptions import MediaException
from pyglet.util import debug_print
def is_corked(self) -> bool:
    assert self._pa_stream is not None
    r = pa.pa_stream_is_corked(self._pa_stream)
    self.context().check(r)
    return bool(r)