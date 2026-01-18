from .charset import MBLENGTH
from .constants import FIELD_TYPE, SERVER_STATUS
from . import err
import struct
import sys
def read_uint24(self):
    low, high = struct.unpack_from('<HB', self._data, self._position)
    self._position += 3
    return low + (high << 16)