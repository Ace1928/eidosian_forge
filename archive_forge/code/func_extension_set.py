import ctypes
from collections import defaultdict
import pyglet
from pyglet.input.base import DeviceOpenException
from pyglet.input.base import Tablet, TabletCanvas
from pyglet.libs.win32 import libwintab as wintab
from pyglet.util import debug_print
def extension_set(self, extension, tablet_id, control_id, function_id, property_id, value):
    prop = wintab.EXTPROPERTY()
    prop.version = 0
    prop.tabletIndex = tablet_id
    prop.controlIndex = control_id
    prop.functionIndex = function_id
    prop.propertyID = property_id
    prop.reserved = 0
    prop.dataSize = ctypes.sizeof(value)
    prop.data[0] = value.value
    success = lib.WTExtSet(self._context, extension, ctypes.byref(prop))
    if success:
        return True
    return False