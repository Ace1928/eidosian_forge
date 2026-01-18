from __future__ import absolute_import, division, print_function
import abc
import binascii
import os
from base64 import b64encode
from datetime import datetime
from hashlib import sha256
from ansible.module_utils import six
from ansible.module_utils.common.text.converters import to_text
from ansible_collections.community.crypto.plugins.module_utils.crypto.support import convert_relative_to_datetime
from ansible_collections.community.crypto.plugins.module_utils.openssh.utils import (
def get_cert_info_object(key_type):
    if key_type == 'rsa':
        cert_info = OpensshRSACertificateInfo()
    elif key_type == 'dsa':
        cert_info = OpensshDSACertificateInfo()
    elif key_type in ('ecdsa-nistp256', 'ecdsa-nistp384', 'ecdsa-nistp521'):
        cert_info = OpensshECDSACertificateInfo()
    elif key_type == 'ed25519':
        cert_info = OpensshED25519CertificateInfo()
    else:
        raise ValueError('%s is not a valid key type' % key_type)
    return cert_info