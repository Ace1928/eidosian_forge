import ctypes
import sys
from typing import Any, Callable, Dict, Optional, Tuple, TYPE_CHECKING, TypeVar, Union
import weakref
from pyglet.media.drivers.pulse import lib_pulseaudio as pa
from pyglet.media.exceptions import MediaException
from pyglet.util import debug_print
def set_write_callback(self, f: PulseAudioStreamRequestCallback) -> None:
    self._cb_write = pa.pa_stream_request_cb_t(f)
    pa.pa_stream_set_write_callback(self._pa_stream, self._cb_write, None)