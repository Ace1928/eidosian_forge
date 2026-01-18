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
def cryptography_verify_certificate_signature(certificate, signer_public_key):
    """
    Check whether the given X509 certificate object was signed by the given public key object.
    """
    return cryptography_verify_signature(certificate.signature, certificate.tbs_certificate_bytes, certificate.signature_hash_algorithm, signer_public_key)