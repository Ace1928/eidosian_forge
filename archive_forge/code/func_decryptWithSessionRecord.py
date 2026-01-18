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
def decryptWithSessionRecord(self, sessionRecord, cipherText):
    """
        :type sessionRecord: SessionRecord
        :type cipherText: WhisperMessage
        """
    previousStates = sessionRecord.getPreviousSessionStates()
    exceptions = []
    try:
        sessionState = SessionState(sessionRecord.getSessionState())
        plaintext = self.decryptWithSessionState(sessionState, cipherText)
        sessionRecord.setState(sessionState)
        return plaintext
    except InvalidMessageException as e:
        exceptions.append(e)
    for i in range(0, len(previousStates)):
        previousState = previousStates[i]
        try:
            promotedState = SessionState(previousState)
            plaintext = self.decryptWithSessionState(promotedState, cipherText)
            previousStates.pop(i)
            sessionRecord.promoteState(promotedState)
            return plaintext
        except InvalidMessageException as e:
            exceptions.append(e)
    raise InvalidMessageException('No valid sessions', exceptions)