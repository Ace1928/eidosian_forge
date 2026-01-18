import sys
from pyasn1.compat.octets import oct2int, null, ensureString
def bitLength(number):
    return int(number).bit_length()