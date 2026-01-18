import ctypes
import io
import operator
import os
import sys
import weakref
from functools import reduce
from pathlib import Path
from tempfile import NamedTemporaryFile
from . import _check_status, _keepref, cairo, constants, ffi
from .fonts import FontOptions, _encode_string
def get_device_offset(self):
    """Returns the previous device offset set by :meth:`set_device_offset`.

        :returns: ``(x_offset, y_offset)``

        """
    offsets = ffi.new('double[2]')
    cairo.cairo_surface_get_device_offset(self._pointer, offsets + 0, offsets + 1)
    return tuple(offsets)