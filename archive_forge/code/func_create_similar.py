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
def create_similar(self, content, width, height):
    """Create a new surface that is as compatible as possible
        for uploading to and the use in conjunction with this surface.
        For example the new surface will have the same fallback resolution
        and :class:`FontOptions`.
        Generally, the new surface will also use the same backend as other,
        unless that is not possible for some reason.

        Initially the surface contents are all 0
        (transparent if contents have transparency, black otherwise.)

        Use :meth:`create_similar_image` if you need an image surface
        which can be painted quickly to the target surface.

        :param content: the :ref:`CONTENT` string for the new surface.
        :param width: width of the new surface (in device-space units)
        :param height: height of the new surface (in device-space units)
        :type content: str
        :type width: int
        :type height: int
        :returns: A new instance of :class:`Surface` or one of its subclasses.

        """
    return Surface._from_pointer(cairo.cairo_surface_create_similar(self._pointer, content, width, height), incref=False)