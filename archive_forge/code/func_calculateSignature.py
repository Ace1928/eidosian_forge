import os
from .eckeypair import ECKeyPair
from ..invalidkeyexception import InvalidKeyException
import axolotl_curve25519 as _curve
@staticmethod
def calculateSignature(privateSigningKey, message):
    """
        :type privateSigningKey: ECPrivateKey
        :type  message: bytearray
        """
    if privateSigningKey.getType() == Curve.DJB_TYPE:
        rand = os.urandom(64)
        res = _curve.calculateSignature(rand, privateSigningKey.getPrivateKey(), message)
        return res
    else:
        raise InvalidKeyException('Unknown type: %s' % privateSigningKey.getType())