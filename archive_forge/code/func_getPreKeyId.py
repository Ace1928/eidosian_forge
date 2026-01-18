from google.protobuf.message import DecodeError
from .ciphertextmessage import CiphertextMessage
from ..util.byteutil import ByteUtil
from ..ecc.curve import Curve
from ..identitykey import IdentityKey
from .whispermessage import WhisperMessage
from ..invalidversionexception import InvalidVersionException
from ..invalidmessageexception import InvalidMessageException
from ..legacymessageexception import LegacyMessageException
from ..invalidkeyexception import InvalidKeyException
from . import whisperprotos_pb2 as whisperprotos
def getPreKeyId(self):
    return self.preKeyId