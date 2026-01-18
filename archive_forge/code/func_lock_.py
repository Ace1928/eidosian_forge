import ctypes
import sys
from typing import Any, Callable, Dict, Optional, Tuple, TYPE_CHECKING, TypeVar, Union
import weakref
from pyglet.media.drivers.pulse import lib_pulseaudio as pa
from pyglet.media.exceptions import MediaException
from pyglet.util import debug_print
def lock_(self) -> None:
    """Lock the threaded mainloop against events.  Required for all
        calls into PA."""
    assert self._pa_threaded_mainloop is not None
    pa.pa_threaded_mainloop_lock(self._pa_threaded_mainloop)