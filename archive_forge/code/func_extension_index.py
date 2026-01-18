import ctypes
from collections import defaultdict
import pyglet
from pyglet.input.base import DeviceOpenException
from pyglet.input.base import Tablet, TabletCanvas
from pyglet.libs.win32 import libwintab as wintab
from pyglet.util import debug_print
def extension_index(ext):
    """Check if a particular extension exists within the driver."""
    exists = True
    i = 0
    index = 4294967295
    while exists:
        tag = wintab.UINT()
        exists = lib.WTInfoW(wintab.WTI_EXTENSIONS + i, wintab.EXT_TAG, ctypes.byref(tag))
        if tag.value == ext:
            index = i
            break
        i += 1
    if index != 4294967295:
        return index
    return None