from . import storageprotos_pb2 as storageprotos
from ..identitykeypair import IdentityKey, IdentityKeyPair
from ..ratchet.rootkey import RootKey
from ..kdf.hkdf import HKDF
from ..ecc.curve import Curve
from ..ecc.eckeypair import ECKeyPair
from ..ratchet.chainkey import ChainKey
from ..kdf.messagekeys import MessageKeys
def getReceiverChain(self, ECPublickKey_senderEphemeral):
    receiverChains = self.sessionStructure.receiverChains
    index = 0
    for receiverChain in receiverChains:
        chainSenderRatchetKey = Curve.decodePoint(bytearray(receiverChain.senderRatchetKey), 0)
        if chainSenderRatchetKey == ECPublickKey_senderEphemeral:
            return (receiverChain, index)
        index += 1