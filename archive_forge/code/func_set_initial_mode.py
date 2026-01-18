import ctypes
import os
import signal
import struct
import threading
from pyglet.libs.x11 import xlib
from pyglet.util import asbytes
def set_initial_mode(mode):
    _install_restore_mode_child()
    display = xlib.XDisplayString(mode.screen.display._display)
    screen = mode.screen.display.x_screen
    if (display, screen) in _restorable_screens:
        return
    packet = ModePacket(display, screen, mode.width, mode.height, mode.rate)
    os.write(_mode_write_pipe, packet.encode())
    _restorable_screens.add((display, screen))