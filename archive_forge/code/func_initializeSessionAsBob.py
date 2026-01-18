from ..ecc.curve import Curve
from .bobaxolotlparamaters import BobAxolotlParameters
from .aliceaxolotlparameters import AliceAxolotlParameters
from ..kdf.hkdfv3 import HKDFv3
from ..util.byteutil import ByteUtil
from .rootkey import RootKey
from .chainkey import ChainKey
from ..protocol.ciphertextmessage import CiphertextMessage
@staticmethod
def initializeSessionAsBob(sessionState, parameters):
    """
        :type sessionState: SessionState
        :type parameters: BobAxolotlParameters
        """
    sessionState.setSessionVersion(CiphertextMessage.CURRENT_VERSION)
    sessionState.setRemoteIdentityKey(parameters.getTheirIdentityKey())
    sessionState.setLocalIdentityKey(parameters.getOurIdentityKey().getPublicKey())
    secrets = bytearray()
    secrets.extend(RatchetingSession.getDiscontinuityBytes())
    secrets.extend(Curve.calculateAgreement(parameters.getTheirIdentityKey().getPublicKey(), parameters.getOurSignedPreKey().getPrivateKey()))
    secrets.extend(Curve.calculateAgreement(parameters.getTheirBaseKey(), parameters.getOurIdentityKey().getPrivateKey()))
    secrets.extend(Curve.calculateAgreement(parameters.getTheirBaseKey(), parameters.getOurSignedPreKey().getPrivateKey()))
    if parameters.getOurOneTimePreKey() is not None:
        secrets.extend(Curve.calculateAgreement(parameters.getTheirBaseKey(), parameters.getOurOneTimePreKey().getPrivateKey()))
    derivedKeys = RatchetingSession.calculateDerivedKeys(secrets)
    sessionState.setSenderChain(parameters.getOurRatchetKey(), derivedKeys.getChainKey())
    sessionState.setRootKey(derivedKeys.getRootKey())