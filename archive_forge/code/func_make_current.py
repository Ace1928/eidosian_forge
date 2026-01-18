import ctypes
import weakref
from collections import namedtuple
from . import lib_openal as al
from . import lib_alc as alc
from pyglet.util import debug_print
from pyglet.media.exceptions import MediaException
def make_current(self):
    alc.alcMakeContextCurrent(self._al_context)
    self.device.check_context_error('Failed to make context current.')