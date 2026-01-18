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
def get_height(self):
    """Return the width of the surface, in pixels."""
    return cairo.cairo_image_surface_get_height(self._pointer)