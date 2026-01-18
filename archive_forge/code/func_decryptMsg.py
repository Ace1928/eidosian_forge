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
def decryptMsg(self, ciphertext, textMsg=True):
    """
        :type ciphertext: WhisperMessage
        :type textMsg: Bool set this to False if you are decrypting bytes
                       instead of string
        """
    if not self.sessionStore.containsSession(self.recipientId, self.deviceId):
        raise NoSessionException('No session for: %s, %s' % (self.recipientId, self.deviceId))
    sessionRecord = self.sessionStore.loadSession(self.recipientId, self.deviceId)
    plaintext = self.decryptWithSessionRecord(sessionRecord, ciphertext)
    self.sessionStore.storeSession(self.recipientId, self.deviceId, sessionRecord)
    return plaintext