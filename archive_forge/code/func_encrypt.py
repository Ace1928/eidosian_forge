from ..invalidkeyidexception import InvalidKeyIdException
from ..invalidkeyexception import InvalidKeyException
from ..invalidmessageexception import InvalidMessageException
from ..duplicatemessagexception import DuplicateMessageException
from ..nosessionexception import NoSessionException
from ..protocol.senderkeymessage import SenderKeyMessage
from ..sessioncipher import AESCipher
from ..groups.state.senderkeystore import SenderKeyStore
def encrypt(self, paddedPlaintext):
    """
        :type paddedPlaintext: bytes
        """
    try:
        record = self.senderKeyStore.loadSenderKey(self.senderKeyName)
        senderKeyState = record.getSenderKeyState()
        senderKey = senderKeyState.getSenderChainKey().getSenderMessageKey()
        ciphertext = self.getCipherText(senderKey.getIv(), senderKey.getCipherKey(), paddedPlaintext)
        senderKeyMessage = SenderKeyMessage(senderKeyState.getKeyId(), senderKey.getIteration(), ciphertext, senderKeyState.getSigningKeyPrivate())
        senderKeyState.setSenderChainKey(senderKeyState.getSenderChainKey().getNext())
        self.senderKeyStore.storeSenderKey(self.senderKeyName, record)
        return senderKeyMessage.serialize()
    except InvalidKeyIdException as e:
        raise NoSessionException(e)