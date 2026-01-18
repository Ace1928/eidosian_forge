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
def generatePreKeys(start, count):
    """
        Generate a list of PreKeys.  Clients should do this at install time, and
        subsequently any time the list of PreKeys stored on the server runs low.

        PreKey IDs are shorts, so they will eventually be repeated.  Clients should
        store PreKeys in a circular buffer, so that they are repeated as infrequently
        as possible.

        @param start The starting PreKey ID, inclusive.
        @param count The number of PreKeys to generate.
        @return the list of generated PreKeyRecords.
        """
    results = []
    start -= 1
    for i in range(0, count):
        preKeyId = (start + i) % (Medium.MAX_VALUE - 1) + 1
        results.append(PreKeyRecord(preKeyId, Curve.generateKeyPair()))
    return results