from __future__ import absolute_import, division, print_function
import base64
from ansible.module_utils._text import to_bytes, to_text
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.cloudstack import (AnsibleCloudStack, cs_argument_spec,
def present_instance(self, start_vm=True):
    instance = self.get_instance()
    if not instance:
        instance = self.deploy_instance(start_vm=start_vm)
    else:
        instance = self.recover_instance(instance=instance)
        instance = self.update_instance(instance=instance, start_vm=start_vm)
    if instance:
        instance = self.ensure_tags(resource=instance, resource_type='UserVm')
        self.instance = instance
    return instance