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
def heat_decrypt(value, encryption_key=None):
    """Decrypt data that has been encrypted using an older version of Heat.

    Note: the encrypt function returns the function that is needed to
    decrypt the data. The database then stores this. When the data is
    then retrieved (potentially by a later version of Heat) the decrypt
    function must still exist. So whilst it may seem that this function
    is not referenced, it will be referenced from the database.
    """
    encryption_key = str.encode(get_valid_encryption_key(encryption_key))
    auth = base64.b64decode(value)
    AES = algorithms.AES(encryption_key)
    block_size_bytes = AES.block_size // 8
    iv = auth[:block_size_bytes]
    backend = backends.default_backend()
    cipher = Cipher(AES, modes.CFB(iv), backend=backend)
    decryptor = cipher.decryptor()
    return decryptor.update(auth[block_size_bytes:]) + decryptor.finalize()