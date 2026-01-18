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
def load_animation(filename, file=None, decoder=None):
    """Load an animation from a file.

    Currently, the only supported format is GIF.

    :Parameters:
        `filename` : str
            Used to guess the animation format, and to load the file if `file`
            is unspecified.
        `file` : file-like object or None
            File object containing the animation stream.
        `decoder` : ImageDecoder or None
            If unspecified, all decoders that are registered for the filename
            extension are tried.  If none succeed, the exception from the
            first decoder is raised.

    :rtype: Animation
    """
    if decoder:
        return decoder.decode_animation(filename, file)
    else:
        return _codec_registry.decode_animation(filename, file)