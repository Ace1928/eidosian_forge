import ctypes
import os
import signal
import struct
import threading
from pyglet.libs.x11 import xlib
from pyglet.util import asbytes
def free_modes_array(modes, n_modes):
    for i in range(n_modes):
        mode = modes.contents[i]
        if mode.privsize:
            xlib.XFree(mode.private)
    xlib.XFree(modes)