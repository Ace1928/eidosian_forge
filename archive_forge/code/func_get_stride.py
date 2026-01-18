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
def get_stride(self):
    """Return the stride of the image surface in bytes
        (or 0 if surface is not an image surface).

        The stride is the distance in bytes
        from the beginning of one row of the image data
        to the beginning of the next row.

        """
    return cairo.cairo_image_surface_get_stride(self._pointer)