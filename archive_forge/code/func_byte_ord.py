import logging
import struct
def byte_ord(c):
    if not isinstance(c, int):
        c = ord(c)
    return c