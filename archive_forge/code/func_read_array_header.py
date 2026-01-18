from datetime import datetime as _DateTime
import sys
import struct
from .exceptions import BufferFull, OutOfData, ExtraData, FormatError, StackError
from .ext import ExtType, Timestamp
def read_array_header(self):
    ret = self._unpack(EX_READ_ARRAY_HEADER)
    self._consume()
    return ret