import re
import struct
import binascii
def b32hexdecode(s, casefold=False):
    return _b32decode(_b32hexalphabet, s, casefold)