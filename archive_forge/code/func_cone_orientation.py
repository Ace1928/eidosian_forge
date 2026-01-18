from collections import namedtuple, defaultdict
import threading
import weakref
from pyglet.media.devices.base import DeviceFlow
import pyglet
from pyglet.libs.win32.types import *
from pyglet.util import debug_print
from pyglet.media.devices import get_audio_device_manager
from . import lib_xaudio2 as lib
@cone_orientation.setter
def cone_orientation(self, value):
    if self.is_emitter:
        x, y, z = value
        self._emitter.OrientFront.x = x
        self._emitter.OrientFront.y = y
        self._emitter.OrientFront.z = z