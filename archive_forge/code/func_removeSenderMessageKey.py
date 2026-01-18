from ...state import storageprotos_pb2 as storageprotos
from ..ratchet.senderchainkey import SenderChainKey
from ..ratchet.sendermessagekey import SenderMessageKey
from ...ecc.curve import Curve
def removeSenderMessageKey(self, iteration):
    keys = self.senderKeyStateStructure.senderMessageKeys
    result = None
    for i in range(0, len(keys)):
        senderMessageKey = keys[i]
        if senderMessageKey.iteration == iteration:
            result = senderMessageKey
            del keys[i]
            break
    if result is not None:
        return SenderMessageKey(result.iteration, bytearray(result.seed))
    return None