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
def font_variant(self, font=None, size=None, index=None, encoding=None, layout_engine=None):
    """
        Create a copy of this FreeTypeFont object,
        using any specified arguments to override the settings.

        Parameters are identical to the parameters used to initialize this
        object.

        :return: A FreeTypeFont object.
        """
    if font is None:
        try:
            font = BytesIO(self.font_bytes)
        except AttributeError:
            font = self.path
    return FreeTypeFont(font=font, size=self.size if size is None else size, index=self.index if index is None else index, encoding=self.encoding if encoding is None else encoding, layout_engine=layout_engine or self.layout_engine)