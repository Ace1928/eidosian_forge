from collections import OrderedDict
from ctypes import *
import pyglet.lib
from pyglet.util import asbytes, asstr
from pyglet.font.base import FontException
def char_index(self, ft_face, character):
    return self._fontconfig.FcFreeTypeCharIndex(ft_face, ord(character))