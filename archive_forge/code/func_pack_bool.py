import struct
from io import BytesIO
from functools import wraps
import warnings
def pack_bool(self, x):
    if x:
        self.__buf.write(b'\x00\x00\x00\x01')
    else:
        self.__buf.write(b'\x00\x00\x00\x00')