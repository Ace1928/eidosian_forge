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
def get_mipmapped_texture(self):
    if self._current_mipmap_texture:
        return self._current_mipmap_texture
    if not self._have_extension():
        return self.get_texture()
    texture = Texture.create(self.width, self.height, GL_TEXTURE_2D, None)
    if self.anchor_x or self.anchor_y:
        texture.anchor_x = self.anchor_x
        texture.anchor_y = self.anchor_y
    glBindTexture(texture.target, texture.id)
    glTexParameteri(texture.target, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
    if not self.mipmap_data:
        glGenerateMipmap(texture.target)
    glCompressedTexImage2D(texture.target, texture.level, self.gl_format, self.width, self.height, 0, len(self.data), self.data)
    width, height = (self.width, self.height)
    level = 0
    for data in self.mipmap_data:
        width >>= 1
        height >>= 1
        level += 1
        glCompressedTexImage2D(texture.target, level, self.gl_format, width, height, 0, len(data), data)
    glFlush()
    self._current_mipmap_texture = texture
    return texture