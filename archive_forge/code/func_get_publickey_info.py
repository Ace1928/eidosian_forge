from __future__ import absolute_import, division, print_function
import abc
import traceback
from ansible.module_utils import six
from ansible.module_utils.basic import missing_required_lib
from ansible.module_utils.common.text.converters import to_native
from ansible_collections.community.crypto.plugins.module_utils.version import LooseVersion
from ansible_collections.community.crypto.plugins.module_utils.crypto.basic import (
from ansible_collections.community.crypto.plugins.module_utils.crypto.support import (
def get_publickey_info(module, backend, content=None, key=None, prefer_one_fingerprint=False):
    if backend == 'cryptography':
        info = PublicKeyInfoRetrievalCryptography(module, content=content, key=key)
    return info.get_info(prefer_one_fingerprint=prefer_one_fingerprint)