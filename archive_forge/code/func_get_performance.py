from collections import namedtuple, defaultdict
import threading
import weakref
from pyglet.media.devices.base import DeviceFlow
import pyglet
from pyglet.libs.win32.types import *
from pyglet.util import debug_print
from pyglet.media.devices import get_audio_device_manager
from . import lib_xaudio2 as lib
def get_performance(self):
    """Retrieve some basic XAudio2 performance data such as memory usage and source counts."""
    pf = lib.XAUDIO2_PERFORMANCE_DATA()
    self._xaudio2.GetPerformanceData(ctypes.byref(pf))
    return pf