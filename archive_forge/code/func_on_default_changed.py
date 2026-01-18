from collections import namedtuple, defaultdict
import threading
import weakref
from pyglet.media.devices.base import DeviceFlow
import pyglet
from pyglet.libs.win32.types import *
from pyglet.util import debug_print
from pyglet.media.devices import get_audio_device_manager
from . import lib_xaudio2 as lib
def on_default_changed(self, device, flow: DeviceFlow):
    if flow == DeviceFlow.OUTPUT:
        'Callback derived from the Audio Devices to help us determine when the system no longer has output.'
        if device is None:
            assert _debug('Error: Default audio device was removed or went missing.')
            self._dead = True
        elif self._dead:
            assert _debug('Warning: Default audio device added after going missing.')
            self._dead = False