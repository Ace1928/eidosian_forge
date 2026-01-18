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
def frombytes(self, data, decoder_name='raw', *args):
    """
        Loads this image with pixel data from a bytes object.

        This method is similar to the :py:func:`~PIL.Image.frombytes` function,
        but loads data into this image instead of creating a new image object.
        """
    if self.width == 0 or self.height == 0:
        return
    if len(args) == 1 and isinstance(args[0], tuple):
        args = args[0]
    if decoder_name == 'raw' and args == ():
        args = self.mode
    d = _getdecoder(self.mode, decoder_name, args)
    d.setimage(self.im)
    s = d.decode(data)
    if s[0] >= 0:
        msg = 'not enough image data'
        raise ValueError(msg)
    if s[1] != 0:
        msg = 'cannot decode image data'
        raise ValueError(msg)