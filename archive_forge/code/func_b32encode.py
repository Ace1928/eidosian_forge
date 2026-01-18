import re
import struct
import binascii
def b32encode(s):
    return _b32encode(_b32alphabet, s)