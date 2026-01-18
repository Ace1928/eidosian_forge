import struct
from passlib.utils import repeat_string
def repeat_encipher(self, l, r, count):
    """repeatedly apply encipher operation to a block"""
    encipher = self.encipher
    n = 0
    while n < count:
        l, r = encipher(l, r)
        n += 1
    return (l, r)