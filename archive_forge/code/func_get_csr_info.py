from __future__ import absolute_import, division, print_function
import abc
import binascii
import traceback
from ansible.module_utils import six
from ansible.module_utils.basic import missing_required_lib
from ansible.module_utils.common.text.converters import to_native
from ansible_collections.community.crypto.plugins.module_utils.version import LooseVersion
from ansible_collections.community.crypto.plugins.module_utils.crypto.support import (
from ansible_collections.community.crypto.plugins.module_utils.crypto.cryptography_support import (
from ansible_collections.community.crypto.plugins.module_utils.crypto.module_backends.publickey_info import (
def get_csr_info(module, backend, content, validate_signature=True, prefer_one_fingerprint=False):
    if backend == 'cryptography':
        info = CSRInfoRetrievalCryptography(module, content, validate_signature=validate_signature)
    return info.get_info(prefer_one_fingerprint=prefer_one_fingerprint)