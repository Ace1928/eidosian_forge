import re
import struct
import binascii
def b32hexencode(s):
    return _b32encode(_b32hexalphabet, s)