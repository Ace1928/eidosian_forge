from __future__ import absolute_import, division, print_function
import base64
from ansible.module_utils._text import to_bytes, to_text
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.cloudstack import (AnsibleCloudStack, cs_argument_spec,
def get_ssh_keypair(self, key=None, name=None, fail_on_missing=True):
    ssh_key_name = name or self.module.params.get('ssh_key')
    if ssh_key_name is None:
        return
    args = {'domainid': self.get_domain('id'), 'account': self.get_account('name'), 'projectid': self.get_project('id'), 'name': ssh_key_name}
    ssh_key_pairs = self.query_api('listSSHKeyPairs', **args)
    if 'sshkeypair' in ssh_key_pairs:
        return self._get_by_key(key=key, my_dict=ssh_key_pairs['sshkeypair'][0])
    elif fail_on_missing:
        self.module.fail_json(msg='SSH key not found: %s' % ssh_key_name)