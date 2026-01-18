from yowsup.common import YowConstants
from yowsup.layers.protocol_iq.protocolentities import ResultIqProtocolEntity
from yowsup.structs import ProtocolTreeNode
from axolotl.state.prekeybundle import PreKeyBundle
from axolotl.identitykey import IdentityKey
from axolotl.ecc.curve import Curve
from axolotl.ecc.djbec import DjbECPublicKey
import binascii
import sys
@staticmethod
def encStr(string):
    if sys.version_info >= (3, 0) and type(string) is str:
        return string.encode('latin-1')
    return string