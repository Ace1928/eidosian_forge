import base64
from hashlib import sha256
import hmac
import binascii
from six import text_type, binary_type
def convert_to_bytes(string_or_bytes):
    if string_or_bytes is None:
        return None
    if isinstance(string_or_bytes, text_type):
        return string_or_bytes.encode('utf-8')
    elif isinstance(string_or_bytes, binary_type):
        return string_or_bytes
    else:
        raise TypeError('Must be a string or bytes object.')