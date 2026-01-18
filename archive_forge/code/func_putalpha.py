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
def putalpha(self, alpha):
    """
        Adds or replaces the alpha layer in this image.  If the image
        does not have an alpha layer, it's converted to "LA" or "RGBA".
        The new layer must be either "L" or "1".

        :param alpha: The new alpha layer.  This can either be an "L" or "1"
           image having the same size as this image, or an integer or
           other color value.
        """
    self._ensure_mutable()
    if self.mode not in ('LA', 'PA', 'RGBA'):
        try:
            mode = getmodebase(self.mode) + 'A'
            try:
                self.im.setmode(mode)
            except (AttributeError, ValueError) as e:
                im = self.im.convert(mode)
                if im.mode not in ('LA', 'PA', 'RGBA'):
                    msg = 'alpha channel could not be added'
                    raise ValueError(msg) from e
                self.im = im
            self.pyaccess = None
            self._mode = self.im.mode
        except KeyError as e:
            msg = 'illegal image mode'
            raise ValueError(msg) from e
    if self.mode in ('LA', 'PA'):
        band = 1
    else:
        band = 3
    if isImageType(alpha):
        if alpha.mode not in ('1', 'L'):
            msg = 'illegal image mode'
            raise ValueError(msg)
        alpha.load()
        if alpha.mode == '1':
            alpha = alpha.convert('L')
    else:
        try:
            self.im.fillband(band, alpha)
        except (AttributeError, ValueError):
            alpha = new('L', self.size, alpha)
        else:
            return
    self.im.putband(alpha.im, band)