from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible.module_utils._text import to_native
from ..module_utils.cloudstack import (
def get_ssh_key(self):
    if not self.ssh_key:
        public_key = self.module.params.get('public_key')
        if public_key:
            args_fingerprint = self._get_common_args()
            args_fingerprint['fingerprint'] = self._get_ssh_fingerprint(public_key)
            ssh_keys = self.query_api('listSSHKeyPairs', **args_fingerprint)
            if ssh_keys and 'sshkeypair' in ssh_keys:
                self.ssh_key = ssh_keys['sshkeypair'][0]
        if not self.ssh_key:
            args_name = self._get_common_args()
            args_name['name'] = self.module.params.get('name')
            ssh_keys = self.query_api('listSSHKeyPairs', **args_name)
            if ssh_keys and 'sshkeypair' in ssh_keys:
                self.ssh_key = ssh_keys['sshkeypair'][0]
    return self.ssh_key