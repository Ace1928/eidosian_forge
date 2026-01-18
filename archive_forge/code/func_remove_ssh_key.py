from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible.module_utils._text import to_native
from ..module_utils.cloudstack import (
def remove_ssh_key(self, name=None):
    ssh_key = self.get_ssh_key()
    if ssh_key:
        self.result['changed'] = True
        args = self._get_common_args()
        args['name'] = name or self.module.params.get('name')
        if not self.module.check_mode:
            self.query_api('deleteSSHKeyPair', **args)
    return ssh_key