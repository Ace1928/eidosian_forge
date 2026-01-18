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
def set_thumbnail_size(self, width, height):
    """Set thumbnail image size for the current and all subsequent pages.

        Setting a width or height of 0 disables thumbnails for the current and
        subsequent pages.

        :param width: thumbnail width.
        :param height: thumbnail height.

        *New in cairo 1.16.*

        *New in cairocffi 0.9.*

        """
    cairo.cairo_pdf_surface_set_thumbnail_size(self._pointer, width, height)