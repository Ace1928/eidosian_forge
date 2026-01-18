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
def create_similar_image(self, content, width, height):
    """
        Create a new image surface that is as compatible as possible
        for uploading to and the use in conjunction with this surface.
        However, this surface can still be used like any normal image surface.

        Initially the surface contents are all 0
        (transparent if contents have transparency, black otherwise.)

        Use :meth:`create_similar` if you don't need an image surface.

        :param format: the :ref:`FORMAT` string for the new surface
        :param width: width of the new surface, (in device-space units)
        :param height: height of the new surface (in device-space units)
        :type format: str
        :type width: int
        :type height: int
        :returns: A new :class:`ImageSurface` instance.

        """
    return Surface._from_pointer(cairo.cairo_surface_create_similar_image(self._pointer, content, width, height), incref=False)