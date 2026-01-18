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
def apply_transparency(self):
    """
        If a P mode image has a "transparency" key in the info dictionary,
        remove the key and instead apply the transparency to the palette.
        Otherwise, the image is unchanged.
        """
    if self.mode != 'P' or 'transparency' not in self.info:
        return
    from . import ImagePalette
    palette = self.getpalette('RGBA')
    transparency = self.info['transparency']
    if isinstance(transparency, bytes):
        for i, alpha in enumerate(transparency):
            palette[i * 4 + 3] = alpha
    else:
        palette[transparency * 4 + 3] = 0
    self.palette = ImagePalette.ImagePalette('RGBA', bytes(palette))
    self.palette.dirty = 1
    del self.info['transparency']