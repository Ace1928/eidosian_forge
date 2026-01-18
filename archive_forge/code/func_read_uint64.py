from .charset import MBLENGTH
from .constants import FIELD_TYPE, SERVER_STATUS
from . import err
import struct
import sys
def read_uint64(self):
    result = struct.unpack_from('<Q', self._data, self._position)[0]
    self._position += 8
    return result