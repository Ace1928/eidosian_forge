import ctypes
import weakref
from collections import namedtuple
from . import lib_openal as al
from . import lib_alc as alc
from pyglet.util import debug_print
from pyglet.media.exceptions import MediaException
@property
def is_paused(self):
    self._get_state()
    return self._state == al.AL_PAUSED