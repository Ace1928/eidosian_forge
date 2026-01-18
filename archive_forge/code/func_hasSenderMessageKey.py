from ...state import storageprotos_pb2 as storageprotos
from ..ratchet.senderchainkey import SenderChainKey
from ..ratchet.sendermessagekey import SenderMessageKey
from ...ecc.curve import Curve
def hasSenderMessageKey(self, iteration):
    for senderMessageKey in self.senderKeyStateStructure.senderMessageKeys:
        if senderMessageKey.iteration == iteration:
            return True
    return False