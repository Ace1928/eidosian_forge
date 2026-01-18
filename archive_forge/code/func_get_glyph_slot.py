import ctypes
import warnings
from collections import namedtuple
from pyglet.util import asbytes, asstr
from pyglet.font import base
from pyglet import image
from pyglet.font.fontconfig import get_fontconfig
from pyglet.font.freetype_lib import *
def get_glyph_slot(self, glyph_index):
    FT_Load_Glyph(self.ft_face, glyph_index, FT_LOAD_RENDER)
    return self.ft_face.contents.glyph.contents