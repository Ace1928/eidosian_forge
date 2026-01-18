from collections import OrderedDict
from ctypes import *
import pyglet.lib
from pyglet.util import asbytes, asstr
from pyglet.font.base import FontException
def create_search_pattern(self):
    return FontConfigSearchPattern(self._fontconfig)