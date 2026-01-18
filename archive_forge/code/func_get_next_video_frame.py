import os
import platform
import warnings
from pyglet import image
from pyglet.libs.win32 import _kernel32 as kernel32
from pyglet.libs.win32 import _ole32 as ole32
from pyglet.libs.win32 import com
from pyglet.libs.win32.constants import *
from pyglet.libs.win32.types import *
from pyglet.media import Source
from pyglet.media.codecs import AudioFormat, AudioData, VideoFormat, MediaDecoder, StaticSource
from pyglet.util import debug_print, DecodeException
def get_next_video_frame(self, skip_empty_frame=True):
    video_data_length = DWORD()
    flags = DWORD()
    timestamp = ctypes.c_longlong()
    if self._current_video_sample:
        self._current_video_buffer.Release()
        self._current_video_sample.Release()
    self._current_video_sample = IMFSample()
    self._current_video_buffer = IMFMediaBuffer()
    while True:
        self._source_reader.ReadSample(self._video_stream_index, 0, None, ctypes.byref(flags), ctypes.byref(timestamp), ctypes.byref(self._current_video_sample))
        if flags.value & MF_SOURCE_READERF_CURRENTMEDIATYPECHANGED:
            assert _debug('WMFVideoDecoder: Data is no longer valid.')
            new = IMFMediaType()
            self._source_reader.GetCurrentMediaType(self._video_stream_index, ctypes.byref(new))
            stride = ctypes.c_uint32()
            new.GetUINT32(MF_MT_DEFAULT_STRIDE, ctypes.byref(stride))
            new.Release()
            self._stride = stride.value
        if flags.value & MF_SOURCE_READERF_ENDOFSTREAM:
            self._timestamp = None
            assert _debug('WMFVideoDecoder: End of data from stream source.')
            break
        if not self._current_video_sample:
            assert _debug('WMFVideoDecoder: No sample.')
            continue
        self._current_video_buffer = IMFMediaBuffer()
        self._current_video_sample.ConvertToContiguousBuffer(ctypes.byref(self._current_video_buffer))
        video_data = POINTER(BYTE)()
        self._current_video_buffer.Lock(ctypes.byref(video_data), None, ctypes.byref(video_data_length))
        width = self.video_format.width
        height = self.video_format.height
        self._timestamp = timestamp_from_wmf(timestamp.value)
        self._current_video_buffer.Unlock()
        return image.ImageData(width, height, 'BGRA', video_data, self._stride)
    return None