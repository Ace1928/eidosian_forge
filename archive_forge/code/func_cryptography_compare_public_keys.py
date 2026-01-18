from __future__ import absolute_import, division, print_function
import base64
import binascii
import re
import sys
import traceback
from ansible.module_utils.common.text.converters import to_text, to_bytes, to_native
from ansible.module_utils.six.moves.urllib.parse import urlparse, urlunparse, ParseResult
from ._asn1 import serialize_asn1_string_as_der
from ansible_collections.community.crypto.plugins.module_utils.version import LooseVersion
from ansible.module_utils.basic import missing_required_lib
from .basic import (
from ._objects import (
from ._obj2txt import obj2txt
def cryptography_compare_public_keys(key1, key2):
    """Tests whether two public keys are the same.

    Needs special logic for Ed25519 and Ed448 keys, since they do not have public_numbers().
    """
    if CRYPTOGRAPHY_HAS_ED25519:
        res = _compare_public_keys(key1, key2, cryptography.hazmat.primitives.asymmetric.ed25519.Ed25519PublicKey)
        if res is not None:
            return res
    if CRYPTOGRAPHY_HAS_ED448:
        res = _compare_public_keys(key1, key2, cryptography.hazmat.primitives.asymmetric.ed448.Ed448PublicKey)
        if res is not None:
            return res
    return key1.public_numbers() == key2.public_numbers()