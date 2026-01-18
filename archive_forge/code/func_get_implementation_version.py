import ctypes
from collections import defaultdict
import pyglet
from pyglet.input.base import DeviceOpenException
from pyglet.input.base import Tablet, TabletCanvas
from pyglet.libs.win32 import libwintab as wintab
from pyglet.util import debug_print
def get_implementation_version():
    impl_version = wtinfo_word(wintab.WTI_INTERFACE, wintab.IFC_IMPLVERSION)
    return impl_version