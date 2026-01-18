from __future__ import absolute_import, division, print_function
import base64
from ansible.module_utils._text import to_bytes, to_text
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.cloudstack import (AnsibleCloudStack, cs_argument_spec,
def restart_instance(self):
    instance = self.get_instance()
    if instance:
        if instance['state'].lower() in ['running', 'starting']:
            self.result['changed'] = True
            if not self.module.check_mode:
                instance = self.query_api('rebootVirtualMachine', id=instance['id'])
                poll_async = self.module.params.get('poll_async')
                if poll_async:
                    instance = self.poll_job(instance, 'virtualmachine')
        elif instance['state'].lower() in ['stopping', 'stopped']:
            instance = self.start_instance()
    return instance