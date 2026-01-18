import queue
import atexit
import weakref
import tempfile
from threading import Event, Thread
from pyglet.util import DecodeException
from .base import StreamingSource, AudioData, AudioFormat, StaticSource
from . import MediaEncoder, MediaDecoder
def pad_added(self, element, pad):
    """pad-added callback"""
    name = pad.query_caps(None).to_string()
    if name.startswith('audio/x-raw'):
        nextpad = self.source.converter.get_static_pad('sink')
        if not nextpad.is_linked():
            self.source.pads = True
            pad.link(nextpad)