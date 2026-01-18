import unittest
from ..state.sessionrecord import SessionRecord
from ..ecc.curve import Curve
from ..identitykeypair import IdentityKeyPair, IdentityKey
from ..ratchet.aliceaxolotlparameters import AliceAxolotlParameters
from ..ratchet.bobaxolotlparamaters import BobAxolotlParameters
from ..ratchet.ratchetingsession import RatchetingSession
from ..tests.inmemoryaxolotlstore import InMemoryAxolotlStore
from ..sessioncipher import SessionCipher
from ..protocol.whispermessage import WhisperMessage
def test_basicSessionV3(self):
    aliceSessionRecord = SessionRecord()
    bobSessionRecord = SessionRecord()
    self.initializeSessionsV3(aliceSessionRecord.getSessionState(), bobSessionRecord.getSessionState())
    self.runInteraction(aliceSessionRecord, bobSessionRecord)