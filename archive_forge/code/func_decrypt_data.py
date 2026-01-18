import base64
import functools
import hashlib
import hmac
import math
import os
from keystonemiddleware.i18n import _
from oslo_utils import secretutils
def decrypt_data(key, data):
    """Decrypt the data with the given secret key."""
    iv = data[:16]
    cipher = ciphers.Cipher(algorithms.AES(key), modes.CBC(iv), backend=crypto_backends.default_backend())
    try:
        decryptor = cipher.decryptor()
        result = decryptor.update(data[16:]) + decryptor.finalize()
    except Exception:
        raise DecryptError(_('Encrypted data appears to be corrupted.'))
    unpadder = padding.PKCS7(algorithms.AES.block_size).unpadder()
    return unpadder.update(result) + unpadder.finalize()