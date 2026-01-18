import time
import weakref
import threading
import pyglet
from pyglet.libs.win32 import com
from pyglet.event import EventDispatcher
from pyglet.libs.win32.types import *
from pyglet.libs.win32 import _ole32 as ole32, _oleaut32 as oleaut32
from pyglet.libs.win32.constants import CLSCTX_INPROC_SERVER
from pyglet.input.base import Device, Controller, Button, AbsoluteAxis, ControllerManager
def rumble_play_weak(self, strength=1.0, duration=0.5):
    self.device.vibration.wRightMotorSpeed = int(max(min(1.0, strength), 0) * 65535)
    self.device.weak_duration = duration
    self.device.set_rumble_state()