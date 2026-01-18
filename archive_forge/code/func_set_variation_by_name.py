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
def set_variation_by_name(self, name):
    """
        :param name: The name of the style.
        :exception OSError: If the font is not a variation font.
        """
    names = self.get_variation_names()
    if not isinstance(name, bytes):
        name = name.encode()
    index = names.index(name) + 1
    if index == getattr(self, '_last_variation_index', None):
        return
    self._last_variation_index = index
    self.font.setvarname(index)