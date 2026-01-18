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
def blit_tiled(self, x, y, z, width, height):
    """Blit this texture tiled over the given area.

        The image will be tiled with the bottom-left corner of the destination
        rectangle aligned with the anchor point of this texture.
        """
    u1 = self.anchor_x / self.width
    v1 = self.anchor_y / self.height
    u2 = u1 + width / self.width
    v2 = v1 + height / self.height
    w, h = (width, height)
    t = self.tex_coords
    vertices = (x, y, z, x + w, y, z, x + w, y + h, z, x, y + h, z)
    tex_coords = (u1, v1, t[2], u2, v1, t[5], u2, v2, t[8], u1, v2, t[11])
    glActiveTexture(GL_TEXTURE0)
    glBindTexture(self.target, self.id)
    pyglet.graphics.draw_indexed(4, GL_TRIANGLES, [0, 1, 2, 0, 2, 3], position=('f', vertices), tex_coords=('f', tex_coords))
    glBindTexture(self.target, 0)