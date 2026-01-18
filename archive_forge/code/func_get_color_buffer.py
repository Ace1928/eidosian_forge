import re
import weakref
from ctypes import *
from io import open, BytesIO
import pyglet
from pyglet.gl import *
from pyglet.gl import gl_info
from pyglet.util import asbytes
from .codecs import ImageEncodeException, ImageDecodeException
from .codecs import registry as _codec_registry
from .codecs import add_default_codecs as _add_default_codecs
from .animation import Animation, AnimationFrame
from .buffer import *
from . import atlas
def get_color_buffer(self):
    """Get the color buffer.

        :rtype: :py:class:`~pyglet.image.ColorBufferImage`
        """
    viewport = self.get_viewport()
    viewport_width = viewport[2]
    viewport_height = viewport[3]
    if not self._color_buffer or viewport_width != self._color_buffer.width or viewport_height != self._color_buffer.height:
        self._color_buffer = ColorBufferImage(*viewport)
    return self._color_buffer