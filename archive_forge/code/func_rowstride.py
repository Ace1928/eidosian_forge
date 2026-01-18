from ctypes import *
from pyglet.gl import *
from pyglet.image import *
from pyglet.image.codecs import *
from pyglet.image.codecs import gif
import pyglet.lib
import pyglet.window
@property
def rowstride(self):
    assert self._pixbuf is not None
    return gdkpixbuf.gdk_pixbuf_get_rowstride(self._pixbuf)