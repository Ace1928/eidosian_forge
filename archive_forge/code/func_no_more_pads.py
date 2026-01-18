import queue
import atexit
import weakref
import tempfile
from threading import Event, Thread
from pyglet.util import DecodeException
from .base import StreamingSource, AudioData, AudioFormat, StaticSource
from . import MediaEncoder, MediaDecoder
def no_more_pads(self, element):
    """Finished Adding pads"""
    if not self.source.pads:
        raise GStreamerDecodeException('No Streams Found')