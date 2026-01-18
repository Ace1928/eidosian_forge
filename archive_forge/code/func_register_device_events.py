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
def register_device_events(self, skip_warning=False, window=None):
    """Register the first OS Window to receive events of disconnect and connection of devices.
        Returns True if events were successfully registered.
        """
    if not self.registered:
        if not window:
            window = pyglet.gl._shadow_window
            if not window:
                for window in pyglet.app.windows:
                    break
        self.window = window
        if self.window is not None:
            dbi = DEV_BROADCAST_DEVICEINTERFACE()
            dbi.dbcc_size = ctypes.sizeof(dbi)
            dbi.dbcc_devicetype = DBT_DEVTYP_DEVICEINTERFACE
            dbi.dbcc_classguid = GUID_DEVINTERFACE_HID
            self._devnotify = _user32.RegisterDeviceNotificationW(self.window._hwnd, ctypes.byref(dbi), DEVICE_NOTIFY_WINDOW_HANDLE)
            self.window._event_handlers[WM_DEVICECHANGE] = self._event_devicechange
            self.registered = True
            self.window.push_handlers(self)
            return True
        elif not skip_warning:
            warnings.warn('DirectInput Device Manager requires a window to receive device connection events.')
    return False