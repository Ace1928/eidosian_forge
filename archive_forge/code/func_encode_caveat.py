import base64
import json
from ._versions import (VERSION_1, VERSION_2, VERSION_3)
from ._third_party import legacy_namespace, ThirdPartyCaveatInfo
from ._keys import PublicKey
from ._error import VerificationError
import macaroonbakery.checkers as checkers
import nacl.public
import six
def encode_caveat(condition, root_key, third_party_info, key, ns):
    """Encrypt a third-party caveat.

    The third_party_info key holds information about the
    third party we're encrypting the caveat for; the key is the
    public/private key pair of the party that's adding the caveat.

    The caveat will be encoded according to the version information
    found in third_party_info.

    @param condition string
    @param root_key bytes
    @param third_party_info object
    @param key nacl key
    @param ns not used yet
    @return bytes
    """
    if third_party_info.version == VERSION_1:
        return _encode_caveat_v1(condition, root_key, third_party_info.public_key, key)
    if third_party_info.version == VERSION_2 or third_party_info.version == VERSION_3:
        return _encode_caveat_v2_v3(third_party_info.version, condition, root_key, third_party_info.public_key, key, ns)
    raise NotImplementedError('only bakery v1, v2, v3 supported')