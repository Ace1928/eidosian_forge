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
def getbbox(self, *, alpha_only=True):
    """
        Calculates the bounding box of the non-zero regions in the
        image.

        :param alpha_only: Optional flag, defaulting to ``True``.
           If ``True`` and the image has an alpha channel, trim transparent pixels.
           Otherwise, trim pixels when all channels are zero.
           Keyword-only argument.
        :returns: The bounding box is returned as a 4-tuple defining the
           left, upper, right, and lower pixel coordinate. See
           :ref:`coordinate-system`. If the image is completely empty, this
           method returns None.

        """
    self.load()
    return self.im.getbbox(alpha_only)