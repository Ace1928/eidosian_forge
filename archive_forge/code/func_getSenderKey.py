from ..invalidkeyidexception import InvalidKeyIdException
from ..invalidkeyexception import InvalidKeyException
from ..invalidmessageexception import InvalidMessageException
from ..duplicatemessagexception import DuplicateMessageException
from ..nosessionexception import NoSessionException
from ..protocol.senderkeymessage import SenderKeyMessage
from ..sessioncipher import AESCipher
from ..groups.state.senderkeystore import SenderKeyStore
def getSenderKey(self, senderKeyState, iteration):
    senderChainKey = senderKeyState.getSenderChainKey()
    if senderChainKey.getIteration() > iteration:
        if senderKeyState.hasSenderMessageKey(iteration):
            return senderKeyState.removeSenderMessageKey(iteration)
        else:
            raise DuplicateMessageException('Received message with old counter: %s, %s' % (senderChainKey.getIteration(), iteration))
    if senderChainKey.getIteration() - iteration > 2000:
        raise InvalidMessageException('Over 2000 messages into the future!')
    while senderChainKey.getIteration() < iteration:
        senderKeyState.addSenderMessageKey(senderChainKey.getSenderMessageKey())
        senderChainKey = senderChainKey.getNext()
    senderKeyState.setSenderChainKey(senderChainKey.getNext())
    return senderChainKey.getSenderMessageKey()