import struct
from io import BytesIO
from functools import wraps
import warnings
def pack_fstring(self, n, s):
    if n < 0:
        raise ValueError('fstring size must be nonnegative')
    data = s[:n]
    n = (n + 3) // 4 * 4
    data = data + (n - len(data)) * b'\x00'
    self.__buf.write(data)