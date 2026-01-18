import os
from .eckeypair import ECKeyPair
from ..invalidkeyexception import InvalidKeyException
import axolotl_curve25519 as _curve
@staticmethod
def decodePrivatePoint(_bytes):
    from .djbec import DjbECPrivateKey
    return DjbECPrivateKey(bytes(_bytes))