import os
from .eckeypair import ECKeyPair
from ..invalidkeyexception import InvalidKeyException
import axolotl_curve25519 as _curve
@staticmethod
def decodePoint(_bytes, offset=0):
    type = _bytes[0]
    if type == Curve.DJB_TYPE:
        from .djbec import DjbECPublicKey
        type = _bytes[offset] & 255
        if type != Curve.DJB_TYPE:
            raise InvalidKeyException('Unknown key type: %s ' % type)
        keyBytes = _bytes[offset + 1:][:32]
        return DjbECPublicKey(bytes(keyBytes))
    else:
        raise InvalidKeyException('Unknown key type: %s' % type)