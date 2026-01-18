import os
from .eckeypair import ECKeyPair
from ..invalidkeyexception import InvalidKeyException
import axolotl_curve25519 as _curve
@staticmethod
def generateKeyPair():
    from .djbec import DjbECPublicKey, DjbECPrivateKey
    privateKey = Curve.generatePrivateKey()
    publicKey = Curve.generatePublicKey(privateKey)
    return ECKeyPair(DjbECPublicKey(publicKey), DjbECPrivateKey(privateKey))