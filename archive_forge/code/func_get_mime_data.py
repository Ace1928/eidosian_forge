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
def get_mime_data(self, mime_type):
    """Return mime data previously attached to surface
        using the specified mime type.

        :param str mime_type: The MIME type of the image data.
        :returns:
            A CFFI buffer object, or :obj:`None`
            if no data has been attached with the given mime type.

        *New in cairo 1.10.*

        """
    buffer_address = ffi.new('unsigned char **')
    buffer_length = ffi.new('unsigned long *')
    mime_type = ffi.new('char[]', mime_type.encode('utf8'))
    cairo.cairo_surface_get_mime_data(self._pointer, mime_type, buffer_address, buffer_length)
    return ffi.buffer(buffer_address[0], buffer_length[0]) if buffer_address[0] != ffi.NULL else None