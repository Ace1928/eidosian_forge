import ctypes
import warnings
from typing import List, Dict, Optional
from pyglet.libs.win32.constants import WM_DEVICECHANGE, DBT_DEVICEARRIVAL, DBT_DEVICEREMOVECOMPLETE, \
from pyglet.event import EventDispatcher
import pyglet
from pyglet.input import base
from pyglet.libs import win32
from pyglet.libs.win32 import dinput, _user32, DEV_BROADCAST_DEVICEINTERFACE, com, DEV_BROADCAST_HDR
from pyglet.libs.win32 import _kernel32
from pyglet.input.controller import get_mapping
from pyglet.input.base import ControllerManager
def get_joysticks(display=None):
    if not _di_manager.registered:
        _di_manager.register_device_events()
        _di_manager.set_current_devices()
    return [joystick for joystick in [_create_joystick(device) for device in _di_manager.devices] if joystick is not None]