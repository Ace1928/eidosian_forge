from ctypes import *
from .base import FontException
import pyglet.lib
def ft_get_library():
    global _library
    if not _library:
        _library = FT_Library()
        error = FT_Init_FreeType(byref(_library))
        if error:
            raise FontException('an error occurred during library initialization', error)
    return _library