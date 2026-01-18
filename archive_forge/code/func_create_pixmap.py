import os
import time
import pytest
import xcffib.xproto  # noqa isort:skip
from xcffib.xproto import ConfigWindow, CW, EventMask, GC  # noqa isort:skip
from . import Context, XCBSurface, cairo_version  # noqa isort:skip
def create_pixmap(conn, wid, width, height):
    """Creates a window of the given dimensions and returns the XID"""
    pixmap = conn.generate_id()
    default_screen = conn.setup.roots[conn.pref_screen]
    conn.core.CreatePixmap(default_screen.root_depth, pixmap, wid, width, height)
    return pixmap