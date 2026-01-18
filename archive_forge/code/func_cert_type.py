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
@cert_type.setter
def cert_type(self, cert_type):
    if cert_type == 'user' or cert_type == _USER_TYPE:
        self._cert_type = _USER_TYPE
    elif cert_type == 'host' or cert_type == _HOST_TYPE:
        self._cert_type = _HOST_TYPE
    else:
        raise ValueError('%s is not a valid certificate type' % cert_type)