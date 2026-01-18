import ctypes
import sys
from typing import Any, Callable, Dict, Optional, Tuple, TYPE_CHECKING, TypeVar, Union
import weakref
from pyglet.media.drivers.pulse import lib_pulseaudio as pa
from pyglet.media.exceptions import MediaException
from pyglet.util import debug_print
def get_sample_spec(self) -> pa.pa_sample_spec:
    assert self._pa_stream is not None
    return pa.pa_stream_get_sample_spec(self._pa_stream)[0]