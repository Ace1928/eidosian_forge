from ..ecc.curve import Curve
from ..kdf.derivedrootsecrets import DerivedRootSecrets
from .chainkey import ChainKey
def getKeyBytes(self):
    return self.key