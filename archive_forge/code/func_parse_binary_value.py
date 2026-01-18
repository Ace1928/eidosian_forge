from array import array
import struct
import sys
import traceback
import types
from Xlib import X
from Xlib.support import lock
def parse_binary_value(self, data, display, length, format):
    values = []
    while 1:
        if len(data) < 2:
            break
        if _bytes_item(data[0]) == 255:
            values.append(struct.unpack('>L', data[1:5])[0])
            data = data[5:]
        elif _bytes_item(data[0]) == 0 and _bytes_item(data[1]) == 0:
            data = data[2:]
        else:
            v, data = self.string_textitem.parse_binary(data, display)
            values.append(v)
    return (values, b'')