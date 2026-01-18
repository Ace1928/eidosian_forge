import base64
from hashlib import sha256
import hmac
import binascii
from six import text_type, binary_type
def hmac_hex(key, data):
    dig = hmac_digest(key, data)
    return binascii.hexlify(dig)