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
def cryptography_compare_private_keys(key1, key2):
    """Tests whether two private keys are the same.

    Needs special logic for Ed25519, X25519, and Ed448 keys, since they do not have private_numbers().
    """
    if CRYPTOGRAPHY_HAS_ED25519:
        res = _compare_private_keys(key1, key2, cryptography.hazmat.primitives.asymmetric.ed25519.Ed25519PrivateKey)
        if res is not None:
            return res
    if CRYPTOGRAPHY_HAS_X25519:
        res = _compare_private_keys(key1, key2, cryptography.hazmat.primitives.asymmetric.x25519.X25519PrivateKey, has_no_private_bytes=not CRYPTOGRAPHY_HAS_X25519_FULL)
        if res is not None:
            return res
    if CRYPTOGRAPHY_HAS_ED448:
        res = _compare_private_keys(key1, key2, cryptography.hazmat.primitives.asymmetric.ed448.Ed448PrivateKey)
        if res is not None:
            return res
    if CRYPTOGRAPHY_HAS_X448:
        res = _compare_private_keys(key1, key2, cryptography.hazmat.primitives.asymmetric.x448.X448PrivateKey)
        if res is not None:
            return res
    return key1.private_numbers() == key2.private_numbers()