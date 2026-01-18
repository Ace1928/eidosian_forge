import hashlib
import hmac
from .sendermessagekey import SenderMessageKey
def getNext(self):
    return SenderChainKey(self.iteration + 1, self.getDerivative(self.__class__.CHAIN_KEY_SEED, self.chainKey))