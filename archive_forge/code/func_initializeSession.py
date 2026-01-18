from ..ecc.curve import Curve
from .bobaxolotlparamaters import BobAxolotlParameters
from .aliceaxolotlparameters import AliceAxolotlParameters
from ..kdf.hkdfv3 import HKDFv3
from ..util.byteutil import ByteUtil
from .rootkey import RootKey
from .chainkey import ChainKey
from ..protocol.ciphertextmessage import CiphertextMessage
@staticmethod
def initializeSession(sessionState, parameters):
    """
        :type sessionState: SessionState
        :type parameters: SymmetricAxolotlParameters
        """
    if RatchetingSession.isAlice(parameters.getOurBaseKey().getPublicKey(), parameters.getTheirBaseKey()):
        aliceParameters = AliceAxolotlParameters.newBuilder()
        aliceParameters.setOurBaseKey(parameters.getOurBaseKey()).setOurIdentityKey(parameters.getOurIdentityKey()).setTheirRatchetKey(parameters.getTheirRatchetKey()).setTheirIdentityKey(parameters.getTheirIdentityKey()).setTheirSignedPreKey(parameters.getTheirBaseKey()).setTheirOneTimePreKey(None)
        RatchetingSession.initializeSessionAsAlice(sessionState, aliceParameters.create())
    else:
        bobParameters = BobAxolotlParameters.newBuilder()
        bobParameters.setOurIdentityKey(parameters.getOurIdentityKey()).setOurRatchetKey(parameters.getOurRatchetKey()).setOurSignedPreKey(parameters.getOurBaseKey()).setOurOneTimePreKey(None).setTheirBaseKey(parameters.getTheirBaseKey()).setTheirIdentityKey(parameters.getTheirIdentityKey())
        RatchetingSession.initializeSessionAsBob(sessionState, bobParameters.create())