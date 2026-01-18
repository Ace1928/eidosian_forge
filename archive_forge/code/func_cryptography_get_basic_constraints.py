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
def cryptography_get_basic_constraints(constraints):
    """
    Given a list of constraints, returns a tuple (ca, path_length).
    Raises an OpenSSLObjectError if a constraint is unknown or cannot be parsed.
    """
    ca = False
    path_length = None
    if constraints:
        for constraint in constraints:
            if constraint.startswith('CA:'):
                if constraint == 'CA:TRUE':
                    ca = True
                elif constraint == 'CA:FALSE':
                    ca = False
                else:
                    raise OpenSSLObjectError('Unknown basic constraint value "{0}" for CA'.format(constraint[3:]))
            elif constraint.startswith('pathlen:'):
                v = constraint[len('pathlen:'):]
                try:
                    path_length = int(v)
                except Exception as e:
                    raise OpenSSLObjectError('Cannot parse path length constraint "{0}" ({1})'.format(v, e))
            else:
                raise OpenSSLObjectError('Unknown basic constraint "{0}"'.format(constraint))
    return (ca, path_length)