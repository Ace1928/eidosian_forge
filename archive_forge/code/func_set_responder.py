import ctypes
import pyglet
from pyglet.input.base import Device, DeviceOpenException
from pyglet.input.base import Button, RelativeAxis, AbsoluteAxis
from pyglet.libs.x11 import xlib
from pyglet.util import asstr
def set_responder(self, device_id, responder):
    self._responders[device_id] = responder