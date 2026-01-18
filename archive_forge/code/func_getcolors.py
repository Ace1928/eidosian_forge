from __future__ import annotations
import atexit
import builtins
import io
import logging
import math
import os
import re
import struct
import sys
import tempfile
import warnings
from collections.abc import Callable, MutableMapping
from enum import IntEnum
from pathlib import Path
from . import (
from ._binary import i32le, o32be, o32le
from ._util import DeferredError, is_path
def getcolors(self, maxcolors=256):
    """
        Returns a list of colors used in this image.

        The colors will be in the image's mode. For example, an RGB image will
        return a tuple of (red, green, blue) color values, and a P image will
        return the index of the color in the palette.

        :param maxcolors: Maximum number of colors.  If this number is
           exceeded, this method returns None.  The default limit is
           256 colors.
        :returns: An unsorted list of (count, pixel) values.
        """
    self.load()
    if self.mode in ('1', 'L', 'P'):
        h = self.im.histogram()
        out = [(h[i], i) for i in range(256) if h[i]]
        if len(out) > maxcolors:
            return None
        return out
    return self.im.getcolors(maxcolors)