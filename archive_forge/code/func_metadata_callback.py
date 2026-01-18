import pyogg
import os.path
import warnings
from abc import abstractmethod
from ctypes import c_void_p, POINTER, c_int, pointer, cast, c_char, c_char_p, CFUNCTYPE, c_ubyte
from ctypes import memmove, create_string_buffer, byref
from pyglet.media import StreamingSource
from pyglet.media.codecs import AudioFormat, AudioData, MediaDecoder, StaticSource
from pyglet.util import debug_print, DecodeException
def metadata_callback(self, decoder, metadata, client_data):
    self.bits_per_sample = metadata.contents.data.stream_info.bits_per_sample
    self.total_samples = metadata.contents.data.stream_info.total_samples
    self.channels = metadata.contents.data.stream_info.channels
    self.frequency = metadata.contents.data.stream_info.sample_rate