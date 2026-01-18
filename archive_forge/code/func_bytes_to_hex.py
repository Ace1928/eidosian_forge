import hashlib
import math
import binascii
from boto.compat import six
def bytes_to_hex(str_as_bytes):
    return binascii.hexlify(str_as_bytes)