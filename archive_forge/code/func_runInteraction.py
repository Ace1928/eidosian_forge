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
def runInteraction(self, aliceStore, bobStore):
    """
        :type aliceStore: AxolotlStore
        :type  bobStore: AxolotlStore
        """
    aliceSessionCipher = SessionCipher(aliceStore, aliceStore, aliceStore, aliceStore, self.__class__.BOB_RECIPIENT_ID, 1)
    bobSessionCipher = SessionCipher(bobStore, bobStore, bobStore, bobStore, self.__class__.ALICE_RECIPIENT_ID, 1)
    originalMessage = b'smert ze smert'
    aliceMessage = aliceSessionCipher.encrypt(originalMessage)
    self.assertTrue(aliceMessage.getType() == CiphertextMessage.WHISPER_TYPE)
    plaintext = bobSessionCipher.decryptMsg(WhisperMessage(serialized=aliceMessage.serialize()))
    self.assertEqual(plaintext, originalMessage)
    bobMessage = bobSessionCipher.encrypt(originalMessage)
    self.assertTrue(bobMessage.getType() == CiphertextMessage.WHISPER_TYPE)
    plaintext = aliceSessionCipher.decryptMsg(WhisperMessage(serialized=bobMessage.serialize()))
    self.assertEqual(plaintext, originalMessage)
    for i in range(0, 10):
        loopingMessage = b'What do we mean by saying that existence precedes essence? We mean that man first of all exists, encounters himself, surges up in the world--and defines himself aftward. %d' % i
        aliceLoopingMessage = aliceSessionCipher.encrypt(loopingMessage)
        loopingPlaintext = bobSessionCipher.decryptMsg(WhisperMessage(serialized=aliceLoopingMessage.serialize()))
        self.assertEqual(loopingPlaintext, loopingMessage)
    for i in range(0, 10):
        loopingMessage = b'What do we mean by saying that existence precedes essence? We mean that man first of all exists, encounters himself, surges up in the world--and defines himself aftward. %d' % i
        bobLoopingMessage = bobSessionCipher.encrypt(loopingMessage)
        loopingPlaintext = aliceSessionCipher.decryptMsg(WhisperMessage(serialized=bobLoopingMessage.serialize()))
        self.assertEqual(loopingPlaintext, loopingMessage)
    aliceOutOfOrderMessages = []
    for i in range(0, 10):
        loopingMessage = b'What do we mean by saying that existence precedes essence? We mean that man first of all exists, encounters himself, surges up in the world--and defines himself aftward. %d' % i
        aliceLoopingMessage = aliceSessionCipher.encrypt(loopingMessage)
        aliceOutOfOrderMessages.append((loopingMessage, aliceLoopingMessage))
    for i in range(0, 10):
        loopingMessage = b'What do we mean by saying that existence precedes essence? We mean that man first of all exists, encounters himself, surges up in the world--and defines himself aftward. %d' % i
        aliceLoopingMessage = aliceSessionCipher.encrypt(loopingMessage)
        loopingPlaintext = bobSessionCipher.decryptMsg(WhisperMessage(serialized=aliceLoopingMessage.serialize()))
        self.assertEqual(loopingPlaintext, loopingMessage)
    for i in range(0, 10):
        loopingMessage = b'You can only desire based on what you know: %d' % i
        bobLoopingMessage = bobSessionCipher.encrypt(loopingMessage)
        loopingPlaintext = aliceSessionCipher.decryptMsg(WhisperMessage(serialized=bobLoopingMessage.serialize()))
        self.assertEqual(loopingPlaintext, loopingMessage)
    for aliceOutOfOrderMessage in aliceOutOfOrderMessages:
        outOfOrderPlaintext = bobSessionCipher.decryptMsg(WhisperMessage(serialized=aliceOutOfOrderMessage[1].serialize()))
        self.assertEqual(outOfOrderPlaintext, aliceOutOfOrderMessage[0])