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
def processKeyExchangeMessage(self, keyExchangeMessage):
    if not self.identityKeyStore.isTrustedIdentity(self.recipientId, keyExchangeMessage.getIdentityKey()):
        raise UntrustedIdentityException(self.recipientId, keyExchangeMessage.getIdentityKey())
    responseMessage = None
    if keyExchangeMessage.isInitiate():
        responseMessage = self.processInitiate(keyExchangeMessage)
    else:
        self.processResponse(keyExchangeMessage)
    return responseMessage