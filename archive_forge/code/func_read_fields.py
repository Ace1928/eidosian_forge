from __future__ import annotations
import io
import os
import struct
from . import Image, ImageFile, _binary
def read_fields(self, field_format):
    size = struct.calcsize(field_format)
    data = self._read_bytes(size)
    return struct.unpack(field_format, data)