from collections import namedtuple, defaultdict
import threading
import weakref
from pyglet.media.devices.base import DeviceFlow
import pyglet
from pyglet.libs.win32.types import *
from pyglet.util import debug_print
from pyglet.media.devices import get_audio_device_manager
from . import lib_xaudio2 as lib
def apply3d(self, source_voice):
    """Apply and immediately commit positional audio effects for the given voice."""
    if self._x3d_handle is not None:
        self._apply3d(source_voice, 2)
        self._xaudio2.CommitChanges(2)