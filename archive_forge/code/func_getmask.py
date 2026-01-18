from __future__ import annotations
import base64
import os
import sys
import warnings
from enum import IntEnum
from io import BytesIO
from pathlib import Path
from typing import BinaryIO
from . import Image
from ._util import is_directory, is_path
def getmask(self, text, mode='', *args, **kwargs):
    im = self.font.getmask(text, mode, *args, **kwargs)
    if self.orientation is not None:
        return im.transpose(self.orientation)
    return im