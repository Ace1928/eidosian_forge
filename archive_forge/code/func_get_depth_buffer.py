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
def get_depth_buffer(self):
    """Get the depth buffer.

        :rtype: :py:class:`~pyglet.image.DepthBufferImage`
        """
    viewport = self.get_viewport()
    viewport_width = viewport[2]
    viewport_height = viewport[3]
    if not self._depth_buffer or viewport_width != self._depth_buffer.width or viewport_height != self._depth_buffer.height:
        self._depth_buffer = DepthBufferImage(*viewport)
    return self._depth_buffer