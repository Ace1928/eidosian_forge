from __future__ import absolute_import, division, print_function
import base64
from ansible.module_utils._text import to_bytes, to_text
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.cloudstack import (AnsibleCloudStack, cs_argument_spec,
def security_groups_has_changed(self):
    security_groups = self.module.params.get('security_groups')
    if security_groups is None:
        return False
    security_groups = [s.lower() for s in security_groups]
    instance_security_groups = self.instance.get('securitygroup') or []
    instance_security_group_names = []
    for instance_security_group in instance_security_groups:
        if instance_security_group['name'].lower() not in security_groups:
            return True
        else:
            instance_security_group_names.append(instance_security_group['name'].lower())
    for security_group in security_groups:
        if security_group not in instance_security_group_names:
            return True
    return False