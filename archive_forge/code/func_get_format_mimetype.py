from __future__ import annotations
import io
import itertools
import struct
import sys
from typing import Any, NamedTuple
from . import Image
from ._deprecate import deprecate
from ._util import is_path
def get_format_mimetype(self):
    if self.custom_mimetype:
        return self.custom_mimetype
    if self.format is not None:
        return Image.MIME.get(self.format.upper())