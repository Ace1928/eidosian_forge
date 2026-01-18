import ctypes
from collections import defaultdict
import pyglet
from pyglet.input.base import DeviceOpenException
from pyglet.input.base import Tablet, TabletCanvas
from pyglet.libs.win32 import libwintab as wintab
from pyglet.util import debug_print
def get_tablet_count():
    """Return just the number of current devices."""
    spec_version = get_spec_version()
    assert _debug(f'Wintab Version: {spec_version}')
    if spec_version < 257:
        return 0
    n_devices = wtinfo_uint(wintab.WTI_INTERFACE, wintab.IFC_NDEVICES)
    return n_devices