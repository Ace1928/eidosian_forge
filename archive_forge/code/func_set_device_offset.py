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
def set_device_offset(self, x_offset, y_offset):
    """ Sets an offset that is added to the device coordinates
        determined by the CTM when drawing to surface.
        One use case for this method is
        when we want to create a :class:`Surface` that redirects drawing
        for a portion of an onscreen surface
        to an offscreen surface in a way that is
        completely invisible to the user of the cairo API.
        Setting a transformation via :meth:`Context.translate`
        isn't sufficient to do this,
        since methods like :meth:`Context.device_to_user`
        will expose the hidden offset.

        Note that the offset affects drawing to the surface
        as well as using the surface in a source pattern.

        :param x_offset:
            The offset in the X direction, in device units
        :param y_offset:
            The offset in the Y direction, in device units

        """
    cairo.cairo_surface_set_device_offset(self._pointer, x_offset, y_offset)
    self._check_status()