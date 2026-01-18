import pyogg
import os.path
import warnings
from abc import abstractmethod
from ctypes import c_void_p, POINTER, c_int, pointer, cast, c_char, c_char_p, CFUNCTYPE, c_ubyte
from ctypes import memmove, create_string_buffer, byref
from pyglet.media import StreamingSource
from pyglet.media.codecs import AudioFormat, AudioData, MediaDecoder, StaticSource
from pyglet.util import debug_print, DecodeException
def tell_callback(self, decoder, offset, data):
    """Decoder wants to know the current position of the file stream."""
    pos = self.file.tell()
    if pos < 0:
        return 1
    else:
        offset.contents.value = pos
        return 0