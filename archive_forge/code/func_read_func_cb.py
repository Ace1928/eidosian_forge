import pyogg
import os.path
import warnings
from abc import abstractmethod
from ctypes import c_void_p, POINTER, c_int, pointer, cast, c_char, c_char_p, CFUNCTYPE, c_ubyte
from ctypes import memmove, create_string_buffer, byref
from pyglet.media import StreamingSource
from pyglet.media.codecs import AudioFormat, AudioData, MediaDecoder, StaticSource
from pyglet.util import debug_print, DecodeException
def read_func_cb(stream, buffer, size):
    data = self.file.read(size)
    read_size = len(data)
    memmove(buffer, data, read_size)
    return read_size