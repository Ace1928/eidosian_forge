import base64
import functools
import hashlib
import hmac
import math
import os
from keystonemiddleware.i18n import _
from oslo_utils import secretutils
@assert_crypto_availability
def encrypt_data(key, data):
    """Encrypt the data with the given secret key.

    Padding is n bytes of the value n, where 1 <= n <= blocksize.
    """
    iv = os.urandom(16)
    cipher = ciphers.Cipher(algorithms.AES(key), modes.CBC(iv), backend=crypto_backends.default_backend())
    padder = padding.PKCS7(algorithms.AES.block_size).padder()
    padded_data = padder.update(data) + padder.finalize()
    encryptor = cipher.encryptor()
    return iv + encryptor.update(padded_data) + encryptor.finalize()