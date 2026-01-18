from __future__ import absolute_import, division, print_function
import base64
from ansible.module_utils._text import to_bytes, to_text
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.cloudstack import (AnsibleCloudStack, cs_argument_spec,
def recover_instance(self, instance):
    if instance['state'].lower() in ['destroying', 'destroyed']:
        self.result['changed'] = True
        if not self.module.check_mode:
            res = self.query_api('recoverVirtualMachine', id=instance['id'])
            instance = res['virtualmachine']
    return instance