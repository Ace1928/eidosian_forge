import logging
from .ecc.curve import Curve
from .ratchet.aliceaxolotlparameters import AliceAxolotlParameters
from .ratchet.bobaxolotlparamaters import BobAxolotlParameters
from .ratchet.symmetricaxolotlparameters import SymmetricAxolotlParameters
from .ratchet.ratchetingsession import RatchetingSession
from .invalidkeyexception import InvalidKeyException
from .invalidkeyidexception import InvalidKeyIdException
from .untrustedidentityexception import UntrustedIdentityException
from .protocol.keyexchangemessage import KeyExchangeMessage
from .protocol.ciphertextmessage import CiphertextMessage
from .statekeyexchangeexception import StaleKeyExchangeException
from .util.medium import Medium
from .util.keyhelper import KeyHelper
def processInitKeyExchangeMessage(self):
    try:
        sequence = KeyHelper.getRandomSequence(65534) + 1
        flags = KeyExchangeMessage.INITIATE_FLAG
        baseKey = Curve.generateKeyPair()
        ratchetKey = Curve.generateKeyPair()
        identityKey = self.identityKeyStore.getIdentityKeyPair()
        baseKeySignature = Curve.calculateSignature(identityKey.getPrivateKey(), baseKey.getPublicKey().serialize())
        sessionRecord = self.sessionStore.loadSession(self.recipientId, self.deviceId)
        sessionRecord.getSessionState().setPendingKeyExchange(sequence, baseKey, ratchetKey, identityKey)
        self.sessionStore.storeSession(self.recipientId, self.deviceId, sessionRecord)
        return KeyExchangeMessage(CiphertextMessage.CURRENT_VERSION, sequence, flags, baseKey.getPublicKey(), baseKeySignature, ratchetKey.getPublicKey(), identityKey.getPublicKey())
    except InvalidKeyException as e:
        raise AssertionError(e)