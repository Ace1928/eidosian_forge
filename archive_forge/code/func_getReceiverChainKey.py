from . import storageprotos_pb2 as storageprotos
from ..identitykeypair import IdentityKey, IdentityKeyPair
from ..ratchet.rootkey import RootKey
from ..kdf.hkdf import HKDF
from ..ecc.curve import Curve
from ..ecc.eckeypair import ECKeyPair
from ..ratchet.chainkey import ChainKey
from ..kdf.messagekeys import MessageKeys
def getReceiverChainKey(self, ECPublicKey_senderEphemeral):
    receiverChainAndIndex = self.getReceiverChain(ECPublicKey_senderEphemeral)
    receiverChain = receiverChainAndIndex[0]
    if receiverChain is None:
        return None
    return ChainKey(HKDF.createFor(self.getSessionVersion()), receiverChain.chainKey.key, receiverChain.chainKey.index)