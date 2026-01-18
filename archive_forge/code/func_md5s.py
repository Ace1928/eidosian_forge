from __future__ import (absolute_import, division, print_function)
import os
from hashlib import sha1
from ansible.errors import AnsibleError
from ansible.module_utils.common.text.converters import to_bytes
def md5s(data):
    if not _md5:
        raise ValueError('MD5 not available.  Possibly running in FIPS mode')
    return secure_hash_s(data, _md5)