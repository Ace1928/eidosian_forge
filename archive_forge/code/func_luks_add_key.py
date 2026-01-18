from __future__ import (absolute_import, division, print_function)
import os
import re
import stat
from ansible.module_utils.basic import AnsibleModule
def luks_add_key(self):
    if self.device is None or (self._module.params['keyfile'] is None and self._module.params['passphrase'] is None) or (self._module.params['new_keyfile'] is None and self._module.params['new_passphrase'] is None):
        return False
    if self._module.params['state'] == 'absent':
        self._module.fail_json(msg='Contradiction in setup: Asking to add a key to absent LUKS.')
    key_present = self._crypthandler.luks_test_key(self.device, self._module.params['new_keyfile'], self._module.params['new_passphrase'])
    if self._module.params['new_keyslot'] is not None:
        key_present_slot = self._crypthandler.luks_test_key(self.device, self._module.params['new_keyfile'], self._module.params['new_passphrase'], self._module.params['new_keyslot'])
        if key_present and (not key_present_slot):
            self._module.fail_json(msg='Trying to add key that is already present in another slot')
    return not key_present