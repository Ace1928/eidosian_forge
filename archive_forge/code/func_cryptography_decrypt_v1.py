import base64
import sys
from cryptography import fernet
from cryptography.hazmat import backends
from cryptography.hazmat.primitives.ciphers import algorithms
from cryptography.hazmat.primitives.ciphers import Cipher
from cryptography.hazmat.primitives.ciphers import modes
from cryptography.hazmat.primitives import padding
from oslo_config import cfg
from oslo_serialization import jsonutils
from oslo_utils import encodeutils
from heat.common import exception
from heat.common.i18n import _
def cryptography_decrypt_v1(value, encryption_key=None):
    encryption_key = get_valid_encryption_key(encryption_key, fix_length=True)
    encoded_key = base64.b64encode(encryption_key.encode('utf-8'))
    sym = fernet.Fernet(encoded_key)
    try:
        return sym.decrypt(encodeutils.safe_encode(value))
    except fernet.InvalidToken:
        raise exception.InvalidEncryptionKey()