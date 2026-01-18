import unittest
import time
import sys
from ..invalidkeyexception import InvalidKeyException
from ..sessionbuilder import SessionBuilder
from ..sessioncipher import SessionCipher
from ..ecc.curve import Curve
from ..protocol.ciphertextmessage import CiphertextMessage
from ..protocol.whispermessage import WhisperMessage
from ..protocol.prekeywhispermessage import PreKeyWhisperMessage
from ..state.prekeybundle import PreKeyBundle
from ..tests.inmemoryaxolotlstore import InMemoryAxolotlStore
from ..state.prekeyrecord import PreKeyRecord
from ..state.signedprekeyrecord import SignedPreKeyRecord
from ..tests.inmemoryidentitykeystore import InMemoryIdentityKeyStore
from ..protocol.keyexchangemessage import KeyExchangeMessage
from ..untrustedidentityexception import UntrustedIdentityException
def test_badSignedPreKeySignature(self):
    aliceStore = InMemoryAxolotlStore()
    aliceSessionBuilder = SessionBuilder(aliceStore, aliceStore, aliceStore, aliceStore, self.__class__.BOB_RECIPIENT_ID, 1)
    bobIdentityKeyStore = InMemoryIdentityKeyStore()
    bobPreKeyPair = Curve.generateKeyPair()
    bobSignedPreKeyPair = Curve.generateKeyPair()
    bobSignedPreKeySignature = Curve.calculateSignature(bobIdentityKeyStore.getIdentityKeyPair().getPrivateKey(), bobSignedPreKeyPair.getPublicKey().serialize())
    for i in range(0, len(bobSignedPreKeySignature) * 8):
        modifiedSignature = bytearray(bobSignedPreKeySignature[:])
        modifiedSignature[int(i / 8)] ^= 1 << i % 8
        bobPreKey = PreKeyBundle(bobIdentityKeyStore.getLocalRegistrationId(), 1, 31337, bobPreKeyPair.getPublicKey(), 22, bobSignedPreKeyPair.getPublicKey(), modifiedSignature, bobIdentityKeyStore.getIdentityKeyPair().getPublicKey())
        try:
            aliceSessionBuilder.processPreKeyBundle(bobPreKey)
        except Exception:
            pass
    bobPreKey = PreKeyBundle(bobIdentityKeyStore.getLocalRegistrationId(), 1, 31337, bobPreKeyPair.getPublicKey(), 22, bobSignedPreKeyPair.getPublicKey(), bobSignedPreKeySignature, bobIdentityKeyStore.getIdentityKeyPair().getPublicKey())
    aliceSessionBuilder.processPreKeyBundle(bobPreKey)