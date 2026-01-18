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
def set_mipmap_image(self, level, image):
    """Set a mipmap image for a particular level.

        The mipmap image will be applied to textures obtained via
        `get_mipmapped_texture`.

        :Parameters:
            `level` : int
                Mipmap level to set image at, must be >= 1.
            `image` : AbstractImage
                Image to set.  Must have correct dimensions for that mipmap
                level (i.e., width >> level, height >> level)
        """
    if level == 0:
        raise ImageException('Cannot set mipmap image at level 0 (it is this image)')
    width, height = (self.width, self.height)
    for i in range(level):
        width >>= 1
        height >>= 1
    if width != image.width or height != image.height:
        raise ImageException('Mipmap image has wrong dimensions for level %d' % level)
    self.mipmap_images += [None] * (level - len(self.mipmap_images))
    self.mipmap_images[level - 1] = image