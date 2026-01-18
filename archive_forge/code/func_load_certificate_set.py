from __future__ import absolute_import, division, print_function
import abc
import base64
import os
import stat
import traceback
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible.module_utils.common.text.converters import to_bytes, to_native
from ansible_collections.community.crypto.plugins.module_utils.version import LooseVersion
from ansible_collections.community.crypto.plugins.module_utils.io import (
from ansible_collections.community.crypto.plugins.module_utils.crypto.basic import (
from ansible_collections.community.crypto.plugins.module_utils.crypto.cryptography_support import (
from ansible_collections.community.crypto.plugins.module_utils.crypto.support import (
from ansible_collections.community.crypto.plugins.module_utils.crypto.pem import (
def load_certificate_set(filename, backend):
    """
    Load list of concatenated PEM files, and return a list of parsed certificates.
    """
    with open(filename, 'rb') as f:
        data = f.read().decode('utf-8')
    return [load_certificate(None, content=cert.encode('utf-8'), backend=backend) for cert in split_pem_list(data)]