from collections import namedtuple, defaultdict
import threading
import weakref
from pyglet.media.devices.base import DeviceFlow
import pyglet
from pyglet.libs.win32.types import *
from pyglet.util import debug_print
from pyglet.media.devices import get_audio_device_manager
from . import lib_xaudio2 as lib
def create_xa2_buffer(audio_data):
    """Creates a XAUDIO2_BUFFER to be used with a source voice.
        Audio data cannot be purged until the source voice has played it; doing so will cause glitches."""
    buff = lib.XAUDIO2_BUFFER()
    buff.AudioBytes = audio_data.length
    buff.pAudioData = ctypes.cast(audio_data.pointer, ctypes.POINTER(ctypes.c_char))
    return buff