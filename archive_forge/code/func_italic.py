from collections import OrderedDict
from ctypes import *
import pyglet.lib
from pyglet.util import asbytes, asstr
from pyglet.font.base import FontException
@property
def italic(self):
    return self._get_integer(FC_SLANT) == FC_SLANT_ITALIC