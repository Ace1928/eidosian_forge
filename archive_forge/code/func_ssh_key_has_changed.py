from __future__ import absolute_import, division, print_function
import base64
from ansible.module_utils._text import to_bytes, to_text
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.cloudstack import (AnsibleCloudStack, cs_argument_spec,
def ssh_key_has_changed(self):
    ssh_key_name = self.module.params.get('ssh_key')
    if ssh_key_name is None:
        return False
    param_ssh_key_fp = self.get_ssh_keypair(key='fingerprint')
    instance_ssh_key_name = self.instance.get('keypair')
    if instance_ssh_key_name is None:
        return True
    instance_ssh_key_fp = self.get_ssh_keypair(key='fingerprint', name=instance_ssh_key_name, fail_on_missing=False)
    if not instance_ssh_key_fp:
        return True
    if instance_ssh_key_fp != param_ssh_key_fp:
        return True
    return False