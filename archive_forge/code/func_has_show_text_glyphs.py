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
def has_show_text_glyphs(self):
    """Returns whether the surface supports sophisticated
        :meth:`Context.show_text_glyphs` operations.
        That is, whether it actually uses the text and cluster data
        provided to a :meth:`Context.show_text_glyphs` call.

        .. note::

            Even if this method returns :obj:`False`,
            :meth:`Context.show_text_glyphs` operation targeted at surface
            will still succeed.
            It just will act like a :meth:`Context.show_glyphs` operation.
            Users can use this method to avoid computing UTF-8 text
            and cluster mapping if the target surface does not use it.

        """
    return bool(cairo.cairo_surface_has_show_text_glyphs(self._pointer))