from collections import namedtuple, defaultdict
import threading
import weakref
from pyglet.media.devices.base import DeviceFlow
import pyglet
from pyglet.libs.win32.types import *
from pyglet.util import debug_print
from pyglet.media.devices import get_audio_device_manager
from . import lib_xaudio2 as lib
@min_distance.setter
def min_distance(self, value):
    if self.is_emitter:
        if self._emitter.CurveDistanceScaler != value:
            self._emitter.CurveDistanceScaler = min(value, lib.FLT_MAX)