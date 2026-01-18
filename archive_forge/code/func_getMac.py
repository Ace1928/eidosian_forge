import hmac
import hashlib
from .ciphertextmessage import CiphertextMessage
from ..util.byteutil import ByteUtil
from ..ecc.curve import Curve
from . import whisperprotos_pb2 as whisperprotos
from ..legacymessageexception import LegacyMessageException
from ..invalidmessageexception import InvalidMessageException
from ..invalidkeyexception import InvalidKeyException
def getMac(self, messageVersion, senderIdentityKey, receiverIdentityKey, macKey, serialized):
    mac = hmac.new(macKey, digestmod=hashlib.sha256)
    if messageVersion >= 3:
        mac.update(senderIdentityKey.getPublicKey().serialize())
        mac.update(receiverIdentityKey.getPublicKey().serialize())
    mac.update(bytes(serialized))
    fullMac = mac.digest()
    return ByteUtil.trim(fullMac, self.__class__.MAC_LENGTH)