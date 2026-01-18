import os.path
from pyglet.util import CodecRegistry, Decoder, Encoder, DecodeException, EncodeException
from pyglet import compat_platform
def decode_animation(self, filename, file):
    """Decode the given file object and return an instance of :py:class:`~pyglet.image.Animation`.
        Throws ImageDecodeException if there is an error.  filename
        can be a file type hint.
        """
    raise ImageDecodeException('This decoder cannot decode animations.')