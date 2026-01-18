from contextlib import contextmanager
from ctypes import create_string_buffer
from enum import IntEnum
import math
from . import ffi
@gname.setter
def gname(self, value):
    if not isinstance(value, bytes):
        value = value.encode(self.header_codec)
    if self.header_codec == 'utf-8':
        ffi.entry_update_gname_utf8(self._entry_p, value)
    else:
        ffi.entry_copy_gname(self._entry_p, value)