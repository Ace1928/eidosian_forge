from __future__ import absolute_import, division
import ctypes.util
import functools
import io
import platform
import struct
import weakref
from .ffi import ffi
def get_extension_item(self, extension, item):
    try:
        _, _, things = next(((k, opcode, v) for k, opcode, v in self.offsets if opcode == extension))
        return things[item]
    except StopIteration:
        raise IndexError(item)