from .charset import MBLENGTH
from .constants import FIELD_TYPE, SERVER_STATUS
from . import err
import struct
import sys
def is_resultset_packet(self):
    field_count = self._data[0]
    return 1 <= field_count <= 250