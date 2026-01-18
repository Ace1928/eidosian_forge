from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.cloudstack import (AnsibleCloudStack, cs_argument_spec,
def get_account_type(self):
    account_type = self.module.params.get('account_type')
    return self.account_types[account_type]