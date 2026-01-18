import ctypes
from collections import defaultdict
import pyglet
from pyglet.input.base import DeviceOpenException
from pyglet.input.base import Tablet, TabletCanvas
from pyglet.libs.win32 import libwintab as wintab
from pyglet.util import debug_print
def on_express_key_release(self, control_id: int, location_id: int):
    """An event called when an ExpressKey is released.

        :Parameters:
            `control_id` : int
                Zero-based index number given to the assigned key by the driver.
                The same control_id may exist in multiple locations, which the location_id is used to differentiate.
            `location_id: int
                Zero-based index indicating side of tablet where control id was found.
                Some tablets may have clusters of ExpressKey's on various areas of the tablet.
                (0 = left, 1 = right, 2 = top, 3 = bottom, 4 = transducer).

        :event:
        """