from typing import List, Optional, Tuple
from pyglet.libs.win32 import com
from pyglet.libs.win32 import _ole32 as ole32
from pyglet.libs.win32.constants import CLSCTX_INPROC_SERVER
from pyglet.libs.win32.types import *
from pyglet.media.devices import base
from pyglet.util import debug_print
def remove_device(self, pwstrDeviceId: str) -> Win32AudioDevice:
    dev = self.audio_devices.get_cached_device(pwstrDeviceId)
    self.audio_devices.devices.remove(dev)
    return dev