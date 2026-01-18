from .ciphertextmessage import CiphertextMessage
from ..util.byteutil import ByteUtil
from ..legacymessageexception import LegacyMessageException
from ..invalidmessageexception import InvalidMessageException
from ..invalidkeyexception import InvalidKeyException
from ..ecc.curve import Curve
from . import whisperprotos_pb2 as whisperprotos
def getSignature(self, signatureKey, serialized):
    """
        :type signatureKey: ECPrivateKey
        :type serialized: bytearray
        """
    try:
        return Curve.calculateSignature(signatureKey, serialized)
    except InvalidKeyException as e:
        raise AssertionError(e)