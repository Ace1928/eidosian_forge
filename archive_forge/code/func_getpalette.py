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
def getpalette(self, rawmode='RGB'):
    """
        Returns the image palette as a list.

        :param rawmode: The mode in which to return the palette. ``None`` will
           return the palette in its current mode.

           .. versionadded:: 9.1.0

        :returns: A list of color values [r, g, b, ...], or None if the
           image has no palette.
        """
    self.load()
    try:
        mode = self.im.getpalettemode()
    except ValueError:
        return None
    if rawmode is None:
        rawmode = mode
    return list(self.im.getpalette(mode, rawmode))