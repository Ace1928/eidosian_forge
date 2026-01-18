import base64
from hashlib import sha256
import hmac
import binascii
from six import text_type, binary_type
def hmac_concat(key, data1, data2):
    hash1 = hmac_digest(key, data1)
    hash2 = hmac_digest(key, data2)
    return hmac_hex(key, hash1 + hash2)