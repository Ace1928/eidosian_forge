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
def generateSignedPreKey(identityKeyPair, signedPreKeyId):
    keyPair = Curve.generateKeyPair()
    signature = Curve.calculateSignature(identityKeyPair.getPrivateKey(), keyPair.getPublicKey().serialize())
    spk = SignedPreKeyRecord(signedPreKeyId, int(round(time.time() * 1000)), keyPair, signature)
    return spk