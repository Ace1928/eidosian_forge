from collections import namedtuple, defaultdict
import threading
import weakref
from pyglet.media.devices.base import DeviceFlow
import pyglet
from pyglet.libs.win32.types import *
from pyglet.util import debug_print
from pyglet.media.devices import get_audio_device_manager
from . import lib_xaudio2 as lib
def flush_on_buffer_end(self, *_):
    if self.voice.buffers_queued == 0:
        self.remaining_data.clear()
        pyglet.clock.schedule_once(self._finish, 0)