import ctypes
import os
import signal
import struct
import threading
from pyglet.libs.x11 import xlib
from pyglet.util import asbytes
def get_matching_mode(modes, n_modes, width, height, rate):
    for i in range(n_modes):
        mode = modes.contents[i]
        if mode.hdisplay == width and mode.vdisplay == height and (mode.dotclock == rate):
            return mode
    return None