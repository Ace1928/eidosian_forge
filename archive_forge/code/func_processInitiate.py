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
def processInitiate(self, keyExchangeMessage):
    flags = KeyExchangeMessage.RESPONSE_FLAG
    sessionRecord = self.sessionStore.loadSession(self.recipientId, self.deviceId)
    if not Curve.verifySignature(keyExchangeMessage.getIdentityKey().getPublicKey(), keyExchangeMessage.getBaseKey().serialize(), keyExchangeMessage.getBaseKeySignature()):
        raise InvalidKeyException('Bad signature!')
    builder = SymmetricAxolotlParameters.newBuilder()
    if not sessionRecord.getSessionState().hasPendingKeyExchange():
        builder.setOurIdentityKey(self.identityKeyStore.getIdentityKeyPair()).setOurBaseKey(Curve.generateKeyPair()).setOurRatchetKey(Curve.generateKeyPair())
    else:
        builder.setOurIdentityKey(sessionRecord.getSessionState().getPendingKeyExchangeIdentityKey()).setOurBaseKey(sessionRecord.getSessionState().getPendingKeyExchangeBaseKey()).setOurRatchetKey(sessionRecord.getSessionState().getPendingKeyExchangeRatchetKey())
        flags |= KeyExchangeMessage.SIMULTAENOUS_INITIATE_FLAG
    builder.setTheirBaseKey(keyExchangeMessage.getBaseKey()).setTheirRatchetKey(keyExchangeMessage.getRatchetKey()).setTheirIdentityKey(keyExchangeMessage.getIdentityKey())
    parameters = builder.create()
    if not sessionRecord.isFresh():
        sessionRecord.archiveCurrentState()
    RatchetingSession.initializeSession(sessionRecord.getSessionState(), parameters)
    self.sessionStore.storeSession(self.recipientId, self.deviceId, sessionRecord)
    self.identityKeyStore.saveIdentity(self.recipientId, keyExchangeMessage.getIdentityKey())
    baseKeySignature = Curve.calculateSignature(parameters.getOurIdentityKey().getPrivateKey(), parameters.getOurBaseKey().getPublicKey().serialize())
    return KeyExchangeMessage(sessionRecord.getSessionState().getSessionVersion(), keyExchangeMessage.getSequence(), flags, parameters.getOurBaseKey().getPublicKey(), baseKeySignature, parameters.getOurRatchetKey().getPublicKey(), parameters.getOurIdentityKey().getPublicKey())