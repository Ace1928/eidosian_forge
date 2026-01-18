from __future__ import absolute_import, division, print_function
import abc
import base64
import traceback
from ansible.module_utils import six
from ansible.module_utils.basic import missing_required_lib
from ansible.module_utils.common.text.converters import to_bytes
from ansible_collections.community.crypto.plugins.module_utils.version import LooseVersion
from ansible_collections.community.crypto.plugins.module_utils.crypto.basic import (
from ansible_collections.community.crypto.plugins.module_utils.crypto.support import (
from ansible_collections.community.crypto.plugins.module_utils.crypto.pem import (
from ansible_collections.community.crypto.plugins.module_utils.crypto.module_backends.privatekey_info import (
from ansible_collections.community.crypto.plugins.module_utils.crypto.module_backends.common import ArgumentSpec
def needs_regeneration(self):
    """Check whether a regeneration is necessary."""
    if self.regenerate == 'always':
        return True
    if not self.has_existing():
        return True
    if not self._check_passphrase():
        if self.regenerate == 'full_idempotence':
            return True
        self.module.fail_json(msg='Unable to read the key. The key is protected with a another passphrase / no passphrase or broken. Will not proceed. To force regeneration, call the module with `generate` set to `full_idempotence` or `always`, or with `force=true`.')
    self._ensure_existing_private_key_loaded()
    if self.regenerate != 'never':
        if not self._check_size_and_type():
            if self.regenerate in ('partial_idempotence', 'full_idempotence'):
                return True
            self.module.fail_json(msg='Key has wrong type and/or size. Will not proceed. To force regeneration, call the module with `generate` set to `partial_idempotence`, `full_idempotence` or `always`, or with `force=true`.')
    if self.format_mismatch == 'regenerate' and self.regenerate != 'never':
        if not self._check_format():
            if self.regenerate in ('partial_idempotence', 'full_idempotence'):
                return True
            self.module.fail_json(msg='Key has wrong format. Will not proceed. To force regeneration, call the module with `generate` set to `partial_idempotence`, `full_idempotence` or `always`, or with `force=true`. To convert the key, set `format_mismatch` to `convert`.')
    return False