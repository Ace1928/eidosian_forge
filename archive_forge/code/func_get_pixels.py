from ctypes import *
from pyglet.gl import *
from pyglet.image import *
from pyglet.image.codecs import *
from pyglet.image.codecs import gif
import pyglet.lib
import pyglet.window
def get_pixels(self):
    pixels = gdkpixbuf.gdk_pixbuf_get_pixels(self._pixbuf)
    assert pixels is not None
    buf = (c_ubyte * (self.rowstride * self.height))()
    memmove(buf, pixels, self.rowstride * (self.height - 1) + self.width * self.channels)
    return buf