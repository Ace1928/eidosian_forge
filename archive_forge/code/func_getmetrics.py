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
def getmetrics(self):
    """
        :return: A tuple of the font ascent (the distance from the baseline to
            the highest outline point) and descent (the distance from the
            baseline to the lowest outline point, a negative value)
        """
    return (self.font.ascent, self.font.descent)