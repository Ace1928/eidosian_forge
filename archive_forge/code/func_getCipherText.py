from ..invalidkeyidexception import InvalidKeyIdException
from ..invalidkeyexception import InvalidKeyException
from ..invalidmessageexception import InvalidMessageException
from ..duplicatemessagexception import DuplicateMessageException
from ..nosessionexception import NoSessionException
from ..protocol.senderkeymessage import SenderKeyMessage
from ..sessioncipher import AESCipher
from ..groups.state.senderkeystore import SenderKeyStore
def getCipherText(self, iv, key, plaintext):
    """
        :type iv: bytearray
        :type key: bytearray
        :type plaintext: bytearray
        """
    cipher = AESCipher(key, iv)
    return cipher.encrypt(plaintext)