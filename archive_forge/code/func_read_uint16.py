from .charset import MBLENGTH
from .constants import FIELD_TYPE, SERVER_STATUS
from . import err
import struct
import sys
def read_uint16(self):
    result = struct.unpack_from('<H', self._data, self._position)[0]
    self._position += 2
    return result