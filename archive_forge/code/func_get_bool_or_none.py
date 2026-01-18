import ctypes
import sys
from typing import Any, Callable, Dict, Optional, Tuple, TYPE_CHECKING, TypeVar, Union
import weakref
from pyglet.media.drivers.pulse import lib_pulseaudio as pa
from pyglet.media.exceptions import MediaException
from pyglet.util import debug_print
def get_bool_or_none(value: int) -> Optional[bool]:
    if value < 0:
        return None
    elif value == 1:
        return True
    else:
        return False