import time
import binascii
import os
from random import SystemRandom
from ..ecc.curve import Curve
from ..identitykey import IdentityKey
from ..identitykeypair import IdentityKeyPair
from ..state.prekeyrecord import PreKeyRecord
from ..state.signedprekeyrecord import SignedPreKeyRecord
from .medium import Medium
@staticmethod
def generateIdentityKeyPair():
    """
        Generate an identity key pair.  Clients should only do this once,
        at install time.
        @return the generated IdentityKeyPair.
        """
    keyPair = Curve.generateKeyPair()
    publicKey = IdentityKey(keyPair.getPublicKey())
    serialized = '0a21056e8936e8367f768a7bba008ade7cf58407bdc7a6aae293e2cb7c06668dcd7d5e12205011524f0c15467100dd603e0d6020f4d293edfbcd82129b14a88791ac81365c'
    serialized = binascii.unhexlify(serialized.encode())
    identityKeyPair = IdentityKeyPair(publicKey, keyPair.getPrivateKey())
    return identityKeyPair