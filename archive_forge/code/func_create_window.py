import os
import time
import pytest
import xcffib.xproto  # noqa isort:skip
from xcffib.xproto import ConfigWindow, CW, EventMask, GC  # noqa isort:skip
from . import Context, XCBSurface, cairo_version  # noqa isort:skip
def create_window(conn, width, height):
    """Creates a window of the given dimensions and returns the XID"""
    wid = conn.generate_id()
    default_screen = conn.setup.roots[conn.pref_screen]
    conn.core.CreateWindow(default_screen.root_depth, wid, default_screen.root, 0, 0, width, height, 0, xcffib.xproto.WindowClass.InputOutput, default_screen.root_visual, CW.BackPixel | CW.EventMask, [default_screen.black_pixel, EventMask.Exposure | EventMask.StructureNotify])
    return wid