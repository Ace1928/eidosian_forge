from __future__ import absolute_import, division, print_function
import base64
import errno
import hashlib
import hmac
import os
import os.path
import re
import tempfile
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_bytes, to_native
def hash_host_key(host, key):
    hmac_key = os.urandom(20)
    hashed_host = hmac.new(hmac_key, to_bytes(host), hashlib.sha1).digest()
    parts = key.strip().split()
    i = 1 if parts[0][0] == '@' else 0
    parts[i] = '|1|%s|%s' % (to_native(base64.b64encode(hmac_key)), to_native(base64.b64encode(hashed_host)))
    return ' '.join(parts)