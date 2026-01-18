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
def restrict_to_version(self, version):
    """Restricts the generated SVG file to ``version``.

        See :meth:`get_versions` for a list of available version values
        that can be used here.

        This method should only be called
        before any drawing operations have been performed on the given surface.
        The simplest way to do this is to call this method
        immediately after creating the surface.

        :param version: A :ref:`SVG_VERSION` string.

        """
    cairo.cairo_svg_surface_restrict_to_version(self._pointer, version)
    self._check_status()