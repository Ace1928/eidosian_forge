import base64
from hashlib import sha256
import hmac
import binascii
from six import text_type, binary_type
def sign_first_party_caveat(signature, predicate):
    return hmac_hex(signature, predicate)