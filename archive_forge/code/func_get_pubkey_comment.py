import hashlib
from libcloud.utils.py3 import b, hexadigits, base64_decode_string
def get_pubkey_comment(pubkey, default=None):
    if pubkey.startswith('ssh-'):
        return pubkey.strip().split(' ', 3)[2]
    if default:
        return default
    raise ValueError('Public key is not in a supported format')