from __future__ import annotations
import io
import os
import struct
from . import Image, ImageFile, _binary
def read_boxes(self):
    size = self.remaining_in_box
    data = self._read_bytes(size)
    return BoxReader(io.BytesIO(data), size)