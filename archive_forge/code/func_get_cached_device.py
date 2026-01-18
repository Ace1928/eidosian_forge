from typing import List, Optional, Tuple
from pyglet.libs.win32 import com
from pyglet.libs.win32 import _ole32 as ole32
from pyglet.libs.win32.constants import CLSCTX_INPROC_SERVER
from pyglet.libs.win32.types import *
from pyglet.media.devices import base
from pyglet.util import debug_print
def get_cached_device(self, dev_id) -> Win32AudioDevice:
    """Gets the cached devices, so we can reduce calls to COM and tell current state vs new states."""
    for device in self.devices:
        if device.id == dev_id:
            return device
    raise Exception('Attempted to get a device that does not exist.', dev_id)