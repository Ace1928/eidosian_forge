import sys
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import padding
from .ecc.curve import Curve
from .sessionbuilder import SessionBuilder
from .state.sessionstate import SessionState
from .protocol.whispermessage import WhisperMessage
from .protocol.prekeywhispermessage import PreKeyWhisperMessage
from .nosessionexception import NoSessionException
from .invalidmessageexception import InvalidMessageException
from .duplicatemessagexception import DuplicateMessageException
import  logging
def getOrCreateMessageKeys(self, sessionState, ECPublicKey_theirEphemeral, chainKey, counter):
    theirEphemeral = ECPublicKey_theirEphemeral
    if chainKey.getIndex() > counter:
        if sessionState.hasMessageKeys(theirEphemeral, counter):
            return sessionState.removeMessageKeys(theirEphemeral, counter)
        else:
            raise DuplicateMessageException('Received message with old counter: %s, %s' % (chainKey.getIndex(), counter))
    if counter - chainKey.getIndex() > 2000:
        raise InvalidMessageException('Over 2000 messages into the future!')
    while chainKey.getIndex() < counter:
        messageKeys = chainKey.getMessageKeys()
        sessionState.setMessageKeys(theirEphemeral, messageKeys)
        chainKey = chainKey.getNextChainKey()
    sessionState.setReceiverChainKey(theirEphemeral, chainKey.getNextChainKey())
    return chainKey.getMessageKeys()