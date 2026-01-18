from collections import OrderedDict
from ctypes import *
import pyglet.lib
from pyglet.util import asbytes, asstr
from pyglet.font.base import FontException
def get_fontconfig():
    global _fontconfig_instance
    if not _fontconfig_instance:
        _fontconfig_instance = FontConfig()
    return _fontconfig_instance