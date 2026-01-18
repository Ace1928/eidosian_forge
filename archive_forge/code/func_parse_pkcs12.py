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
def parse_pkcs12(pkcs12_bytes, passphrase=None):
    """Returns a tuple (private_key, certificate, additional_certificates, friendly_name).
    """
    if _load_pkcs12 is None and _load_key_and_certificates is None:
        raise ValueError('neither load_pkcs12() nor load_key_and_certificates() present in the current cryptography version')
    if passphrase is not None:
        passphrase = to_bytes(passphrase)
    if _load_pkcs12 is not None:
        return _parse_pkcs12_36_0_0(pkcs12_bytes, passphrase)
    if LooseVersion(cryptography.__version__) >= LooseVersion('35.0'):
        return _parse_pkcs12_35_0_0(pkcs12_bytes, passphrase)
    return _parse_pkcs12_legacy(pkcs12_bytes, passphrase)