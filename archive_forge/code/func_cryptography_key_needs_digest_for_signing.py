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
def cryptography_key_needs_digest_for_signing(key):
    """Tests whether the given private key requires a digest algorithm for signing.

    Ed25519 and Ed448 keys do not; they need None to be passed as the digest algorithm.
    """
    if CRYPTOGRAPHY_HAS_ED25519 and isinstance(key, cryptography.hazmat.primitives.asymmetric.ed25519.Ed25519PrivateKey):
        return False
    if CRYPTOGRAPHY_HAS_ED448 and isinstance(key, cryptography.hazmat.primitives.asymmetric.ed448.Ed448PrivateKey):
        return False
    return True