import base64
import json
from ._versions import (VERSION_1, VERSION_2, VERSION_3)
from ._third_party import legacy_namespace, ThirdPartyCaveatInfo
from ._keys import PublicKey
from ._error import VerificationError
import macaroonbakery.checkers as checkers
import nacl.public
import six
def decode_uvarint(data):
    """Decode a variable-length integer.

    Reads a sequence of unsigned integer byte and decodes them into an integer
    in variable-length format and returns it and the length read.
    """
    n = 0
    shift = 0
    length = 0
    for b in data:
        if not isinstance(b, int):
            b = six.byte2int(b)
        n |= (b & 127) << shift
        length += 1
        if b & 128 == 0:
            break
        shift += 7
    return (n, length)