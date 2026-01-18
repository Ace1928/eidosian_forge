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
def set_mime_data(self, mime_type, data):
    """
        Attach an image in the format ``mime_type`` to this surface.

        To remove the data from a surface,
        call this method with same mime type and :obj:`None` for data.

        The attached image (or filename) data can later
        be used by backends which support it
        (currently: PDF, PS, SVG and Win32 Printing surfaces)
        to emit this data instead of making a snapshot of the surface.
        This approach tends to be faster
        and requires less memory and disk space.

        The recognized MIME types are the following:

        ``"image/png"``
            The Portable Network Graphics image file format (ISO/IEC 15948).
        ``"image/jpeg"``
            The Joint Photographic Experts Group (JPEG)
            image coding standard (ISO/IEC 10918-1).
        ``"image/jp2"``
            The Joint Photographic Experts Group (JPEG) 2000
            image coding standard (ISO/IEC 15444-1).
        ``"text/x-uri"``
            URL for an image file (unofficial MIME type).

        See corresponding backend surface docs
        for details about which MIME types it can handle.
        Caution: the associated MIME data will be discarded
        if you draw on the surface afterwards.
        Use this method with care.

        :param str mime_type: The MIME type of the image data.
        :param bytes data: The image data to attach to the surface.

        *New in cairo 1.10.*

        """
    mime_type = ffi.new('char[]', mime_type.encode('utf8'))
    if data is None:
        _check_status(cairo.cairo_surface_set_mime_data(self._pointer, mime_type, ffi.NULL, 0, ffi.NULL, ffi.NULL))
    else:
        length = len(data)
        data = ffi.new('unsigned char[]', data)
        keep_alive = KeepAlive(data, mime_type)
        _check_status(cairo.cairo_surface_set_mime_data(self._pointer, mime_type, data, length, *keep_alive.closure))
        keep_alive.save()