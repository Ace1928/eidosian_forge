import base64
from hashlib import sha256
import hmac
import binascii
from six import text_type, binary_type
def sign_third_party_caveat(signature, verification_id, caveat_id):
    return hmac_concat(signature, verification_id, caveat_id)