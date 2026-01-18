import hmac
import hashlib
from .ciphertextmessage import CiphertextMessage
from ..util.byteutil import ByteUtil
from ..ecc.curve import Curve
from . import whisperprotos_pb2 as whisperprotos
from ..legacymessageexception import LegacyMessageException
from ..invalidmessageexception import InvalidMessageException
from ..invalidkeyexception import InvalidKeyException
def getCounter(self):
    return self.counter