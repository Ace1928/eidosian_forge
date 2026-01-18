import array
import os
import struct
import six
from ._exceptions import *
from ._utils import validate_utf8
from threading import Lock
def recv_header(self):
    header = self.recv_strict(2)
    b1 = header[0]
    if six.PY2:
        b1 = ord(b1)
    fin = b1 >> 7 & 1
    rsv1 = b1 >> 6 & 1
    rsv2 = b1 >> 5 & 1
    rsv3 = b1 >> 4 & 1
    opcode = b1 & 15
    b2 = header[1]
    if six.PY2:
        b2 = ord(b2)
    has_mask = b2 >> 7 & 1
    length_bits = b2 & 127
    self.header = (fin, rsv1, rsv2, rsv3, opcode, has_mask, length_bits)