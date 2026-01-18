import ctypes
import weakref
from collections import namedtuple
from . import lib_openal as al
from . import lib_alc as alc
from pyglet.util import debug_print
from pyglet.media.exceptions import MediaException
def create_source(self):
    self.make_current()
    new_source = OpenALSource(self)
    self._sources.add(new_source)
    return new_source