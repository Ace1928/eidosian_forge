import ctypes
import sys
from typing import Any, Callable, Dict, Optional, Tuple, TYPE_CHECKING, TypeVar, Union
import weakref
from pyglet.media.drivers.pulse import lib_pulseaudio as pa
from pyglet.media.exceptions import MediaException
from pyglet.util import debug_print
def get_timing_info(self) -> Optional[pa.pa_timing_info]:
    """
        Retrieves the stream's timing_info struct,
        or None if it does not exist.
        Note that ctypes creates a copy of the struct, meaning it will
        be safe to use with an unlocked mainloop.
        """
    context = self.context()
    assert context is not None
    assert self._pa_stream is not None
    timing_info = pa.pa_stream_get_timing_info(self._pa_stream)
    return timing_info.contents if timing_info else None