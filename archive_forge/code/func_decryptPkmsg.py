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
def decryptPkmsg(self, ciphertext, textMsg=True):
    """
        :type ciphertext: PreKeyWhisperMessage
        """
    sessionRecord = self.sessionStore.loadSession(self.recipientId, self.deviceId)
    unsignedPreKeyId = self.sessionBuilder.process(sessionRecord, ciphertext)
    plaintext = self.decryptWithSessionRecord(sessionRecord, ciphertext.getWhisperMessage())
    self.sessionStore.storeSession(self.recipientId, self.deviceId, sessionRecord)
    if unsignedPreKeyId is not None:
        self.preKeyStore.removePreKey(unsignedPreKeyId)
    return plaintext