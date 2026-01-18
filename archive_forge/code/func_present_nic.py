from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.cloudstack import (AnsibleCloudStack, cs_argument_spec,
def present_nic(self):
    nic = self.get_nic()
    if not nic:
        nic = self.add_nic()
    else:
        nic = self.update_nic(nic)
    return nic