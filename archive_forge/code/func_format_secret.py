import os
from typing import Any, List, Tuple
from jeepney import (
from jeepney.io.blocking import DBusConnection
from secretstorage.defines import DBUS_UNKNOWN_METHOD, DBUS_NO_SUCH_OBJECT, \
from secretstorage.dhcrypto import Session, int_to_bytes
from secretstorage.exceptions import ItemNotFoundException, \
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
def format_secret(session: Session, secret: bytes, content_type: str) -> Tuple[str, bytes, bytes, str]:
    """Formats `secret` to make possible to pass it to the
    Secret Service API."""
    if isinstance(secret, str):
        secret = secret.encode('utf-8')
    elif not isinstance(secret, bytes):
        raise TypeError('secret must be bytes')
    assert session.object_path is not None
    if not session.encrypted:
        return (session.object_path, b'', secret, content_type)
    assert session.aes_key is not None
    padding = 16 - (len(secret) & 15)
    secret += bytes((padding,) * padding)
    aes_iv = os.urandom(16)
    aes = algorithms.AES(session.aes_key)
    encryptor = Cipher(aes, modes.CBC(aes_iv), default_backend()).encryptor()
    encrypted_secret = encryptor.update(secret) + encryptor.finalize()
    return (session.object_path, aes_iv, encrypted_secret, content_type)