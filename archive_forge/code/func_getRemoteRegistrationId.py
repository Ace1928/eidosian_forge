from . import storageprotos_pb2 as storageprotos
from ..identitykeypair import IdentityKey, IdentityKeyPair
from ..ratchet.rootkey import RootKey
from ..kdf.hkdf import HKDF
from ..ecc.curve import Curve
from ..ecc.eckeypair import ECKeyPair
from ..ratchet.chainkey import ChainKey
from ..kdf.messagekeys import MessageKeys
def getRemoteRegistrationId(self, registrationId):
    return self.sessionStructure.remoteRegistrationId