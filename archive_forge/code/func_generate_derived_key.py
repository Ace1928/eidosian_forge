import base64
from hashlib import sha256
import hmac
import binascii
from six import text_type, binary_type
def generate_derived_key(key):
    return hmac_digest(b'macaroons-key-generator', key)