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
def getOrCreateChainKey(self, sessionState, ECPublickKey_theirEphemeral):
    theirEphemeral = ECPublickKey_theirEphemeral
    if sessionState.hasReceiverChain(theirEphemeral):
        return sessionState.getReceiverChainKey(theirEphemeral)
    else:
        rootKey = sessionState.getRootKey()
        ourEphemeral = sessionState.getSenderRatchetKeyPair()
        receiverChain = rootKey.createChain(theirEphemeral, ourEphemeral)
        ourNewEphemeral = Curve.generateKeyPair()
        senderChain = receiverChain[0].createChain(theirEphemeral, ourNewEphemeral)
        sessionState.setRootKey(senderChain[0])
        sessionState.addReceiverChain(theirEphemeral, receiverChain[1])
        sessionState.setPreviousCounter(max(sessionState.getSenderChainKey().getIndex() - 1, 0))
        sessionState.setSenderChain(ourNewEphemeral, senderChain[1])
        return receiverChain[1]