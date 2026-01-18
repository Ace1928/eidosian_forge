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
def getname(self):
    """
        :return: A tuple of the font family (e.g. Helvetica) and the font style
            (e.g. Bold)
        """
    return (self.font.family, self.font.style)