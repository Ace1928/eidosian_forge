import json
import mmap
import os
import struct
from typing import List
@staticmethod
def read_all_values_from_file(filename):
    with open(filename, 'rb') as infp:
        data = infp.read(mmap.PAGESIZE)
        used = _unpack_integer(data, 0)[0]
        if used > len(data):
            data += infp.read(used - len(data))
    return _read_all_values(data, used)