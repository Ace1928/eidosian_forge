import re
import ctypes
from pyglet.gl import *
from pyglet.gl import gl_info
from pyglet.image import AbstractImage, Texture
def get_texture(self, rectangle=False, force_rectangle=False):
    """The parameters 'rectangle' and 'force_rectangle' are ignored.
           See the documentation of the method 'AbstractImage.get_texture' for
           a more detailed documentation of the method. """
    return self._get_texture()