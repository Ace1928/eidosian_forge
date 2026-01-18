from ...state import storageprotos_pb2 as storageprotos
from ..ratchet.senderchainkey import SenderChainKey
from ..ratchet.sendermessagekey import SenderMessageKey
from ...ecc.curve import Curve
def getSigningKeyPrivate(self):
    return Curve.decodePrivatePoint(self.senderKeyStateStructure.senderSigningKey.private)