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
def decrypted_dict(data, encryption_key=None):
    """Return a decrypted dict. Assume input values are encrypted json fields."""
    return_data = {}
    if not data:
        return return_data
    for prop_name, prop_value in data.items():
        method, value = prop_value
        try:
            decrypted_value = decrypt(method, value, encryption_key)
        except UnicodeDecodeError:
            raise exception.InvalidEncryptionKey()
        prop_string = jsonutils.loads(decrypted_value)
        return_data[prop_name] = prop_string
    return return_data