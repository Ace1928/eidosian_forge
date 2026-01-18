import os
import signal
import struct
import sys
from zope.interface import implementer
from twisted.conch.interfaces import (
from twisted.conch.ssh import channel, common, connection
from twisted.internet import interfaces, protocol
from twisted.logger import Logger
from twisted.python.compat import networkString
def packRequest_window_change(geometry):
    """
    Pack a window-change request so that it is suitable for sending.

    @type geometry: L{tuple}
    @param geometry: A tuple of (rows, columns, xpixel, ypixel)
    """
    rows, cols, xpixel, ypixel = geometry
    return struct.pack('>4L', cols, rows, xpixel, ypixel)