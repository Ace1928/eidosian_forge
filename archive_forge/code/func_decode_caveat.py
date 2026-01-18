import base64
import json
from ._versions import (VERSION_1, VERSION_2, VERSION_3)
from ._third_party import legacy_namespace, ThirdPartyCaveatInfo
from ._keys import PublicKey
from ._error import VerificationError
import macaroonbakery.checkers as checkers
import nacl.public
import six
def decode_caveat(key, caveat):
    """Decode caveat by decrypting the encrypted part using key.

    @param key the nacl private key to decode.
    @param caveat bytes.
    @return ThirdPartyCaveatInfo
    """
    if len(caveat) == 0:
        raise VerificationError('empty third party caveat')
    first = caveat[:1]
    if first == b'e':
        return _decode_caveat_v1(key, caveat)
    first_as_int = six.byte2int(first)
    if first_as_int == VERSION_2 or first_as_int == VERSION_3:
        if len(caveat) < _VERSION3_CAVEAT_MIN_LEN and first_as_int == VERSION_3:
            raise VerificationError('caveat id payload not provided for caveat id {}'.format(caveat))
        return _decode_caveat_v2_v3(first_as_int, key, caveat)
    raise VerificationError('unknown version for caveat')