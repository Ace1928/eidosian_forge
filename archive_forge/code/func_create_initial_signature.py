import base64
from hashlib import sha256
import hmac
import binascii
from six import text_type, binary_type
def create_initial_signature(key, identifier):
    derived_key = generate_derived_key(key)
    return hmac_hex(derived_key, identifier)