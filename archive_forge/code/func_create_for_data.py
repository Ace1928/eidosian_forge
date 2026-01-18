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
@classmethod
def create_for_data(cls, data, format, width, height, stride=None):
    """Same as ``ImageSurface(format, width, height, data, stride)``.
        Exists for compatibility with pycairo.

        """
    return cls(format, width, height, data, stride)