import os
import sys
from enum import Enum, _simple_enum
@property
def bytes_le(self):
    bytes = self.bytes
    return bytes[4 - 1::-1] + bytes[6 - 1:4 - 1:-1] + bytes[8 - 1:6 - 1:-1] + bytes[8:]