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
def getlength(self, text, *args, **kwargs):
    if self.orientation in (Image.Transpose.ROTATE_90, Image.Transpose.ROTATE_270):
        msg = 'text length is undefined for text rotated by 90 or 270 degrees'
        raise ValueError(msg)
    return self.font.getlength(text, *args, **kwargs)