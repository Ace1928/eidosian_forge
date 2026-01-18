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
def oslo_decrypt_v1(value, encryption_key=None):
    encryption_key = get_valid_encryption_key(encryption_key)
    sym = SymmetricCrypto()
    return sym.decrypt(encryption_key, value, b64decode=True)