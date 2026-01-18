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
def cryptography_parse_relative_distinguished_name(rdn):
    names = []
    for part in rdn:
        try:
            names.append(_parse_dn_component(to_bytes(part), decode_remainder=False)[0])
        except OpenSSLObjectError as e:
            raise OpenSSLObjectError(u'Error while parsing relative distinguished name "{0}": {1}'.format(part, e))
    return cryptography.x509.RelativeDistinguishedName(names)