from ctypes import memmove, byref, c_uint32, sizeof, cast, c_void_p, create_string_buffer, POINTER, c_char, \
from pyglet.libs.darwin import cf, CFSTR
from pyglet.libs.darwin.coreaudio import kCFURLPOSIXPathStyle, AudioStreamBasicDescription, ca, ExtAudioFileRef, \
from pyglet.media import StreamingSource, StaticSource
from pyglet.media.codecs import AudioFormat, MediaDecoder, AudioData
def read_data_cb(ref, offset, requested_length, buffer, actual_count):
    self.file.seek(offset)
    data = self.file.read(requested_length)
    data_size = len(data)
    memmove(buffer, data, data_size)
    actual_count.contents.value = data_size
    return 0