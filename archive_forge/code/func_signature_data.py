from __future__ import absolute_import, division, print_function
import os
import re
from contextlib import contextmanager
from struct import Struct
from ansible.module_utils.six import PY3
@classmethod
def signature_data(cls, signature_string):
    signature_data = {}
    parser = cls(signature_string)
    signature_type = parser.string()
    signature_blob = parser.string()
    blob_parser = cls(signature_blob)
    if signature_type in (b'ssh-rsa', b'rsa-sha2-256', b'rsa-sha2-512'):
        signature_data['s'] = cls._big_int(signature_blob, 'big')
    elif signature_type == b'ssh-dss':
        signature_data['r'] = cls._big_int(signature_blob[:20], 'big')
        signature_data['s'] = cls._big_int(signature_blob[20:], 'big')
    elif signature_type in (b'ecdsa-sha2-nistp256', b'ecdsa-sha2-nistp384', b'ecdsa-sha2-nistp521'):
        signature_data['r'] = blob_parser.mpint()
        signature_data['s'] = blob_parser.mpint()
    elif signature_type == b'ssh-ed25519':
        signature_data['R'] = cls._big_int(signature_blob[:32], 'little')
        signature_data['S'] = cls._big_int(signature_blob[32:], 'little')
    else:
        raise ValueError('%s is not a valid signature type' % signature_type)
    signature_data['signature_type'] = signature_type
    return signature_data