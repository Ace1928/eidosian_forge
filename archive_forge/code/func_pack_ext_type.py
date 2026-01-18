from datetime import datetime as _DateTime
import sys
import struct
from .exceptions import BufferFull, OutOfData, ExtraData, FormatError, StackError
from .ext import ExtType, Timestamp
def pack_ext_type(self, typecode, data):
    if not isinstance(typecode, int):
        raise TypeError('typecode must have int type.')
    if not 0 <= typecode <= 127:
        raise ValueError('typecode should be 0-127')
    if not isinstance(data, bytes):
        raise TypeError('data must have bytes type')
    L = len(data)
    if L > 4294967295:
        raise ValueError('Too large data')
    if L == 1:
        self._buffer.write(b'\xd4')
    elif L == 2:
        self._buffer.write(b'\xd5')
    elif L == 4:
        self._buffer.write(b'\xd6')
    elif L == 8:
        self._buffer.write(b'\xd7')
    elif L == 16:
        self._buffer.write(b'\xd8')
    elif L <= 255:
        self._buffer.write(b'\xc7' + struct.pack('B', L))
    elif L <= 65535:
        self._buffer.write(b'\xc8' + struct.pack('>H', L))
    else:
        self._buffer.write(b'\xc9' + struct.pack('>I', L))
    self._buffer.write(struct.pack('B', typecode))
    self._buffer.write(data)