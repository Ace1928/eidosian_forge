from __future__ import (absolute_import, division, print_function)
import os.path
import tempfile
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.urls import fetch_url
from ansible.module_utils.common.text.converters import to_native
def sanitise_keyid(self, keyid):
    """Sanitise given key ID.

        Strips whitespace, uppercases all characters, and strips leading `0X`.
        """
    sanitised_keyid = keyid.strip().upper().replace(' ', '').replace('0X', '')
    if len(sanitised_keyid) != self.keylength:
        self.module.fail_json(msg='key ID is not full-length: %s' % sanitised_keyid)
    if not self.is_hexadecimal(sanitised_keyid):
        self.module.fail_json(msg='key ID is not hexadecimal: %s' % sanitised_keyid)
    return sanitised_keyid