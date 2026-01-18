import hmac
import hashlib
from ..kdf.derivedmessagesecrets import DerivedMessageSecrets
from ..kdf.messagekeys import MessageKeys
def getNextChainKey(self):
    nextKey = self.getBaseMaterial(self.__class__.CHAIN_KEY_SEED)
    return ChainKey(self.kdf, nextKey, self.index + 1)