from ..ecc.curve import Curve
from .bobaxolotlparamaters import BobAxolotlParameters
from .aliceaxolotlparameters import AliceAxolotlParameters
from ..kdf.hkdfv3 import HKDFv3
from ..util.byteutil import ByteUtil
from .rootkey import RootKey
from .chainkey import ChainKey
from ..protocol.ciphertextmessage import CiphertextMessage
def getChainKey(self):
    return self.chainKey