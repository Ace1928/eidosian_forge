from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import (
from ansible.module_utils._text import to_native
def check_uplinks(self):
    pnic_device = []
    self.set_desired_state()
    for dvs_host_member in self.dv_switch.config.host:
        if dvs_host_member.config.host.name == self.esxi_hostname:
            break
    for pnicSpec in dvs_host_member.config.backing.pnicSpec:
        pnic_device.append(pnicSpec.pnicDevice)
        if pnicSpec.pnicDevice not in self.desired_state:
            return False
        if pnicSpec.uplinkPortKey != self.desired_state[pnicSpec.pnicDevice]:
            return False
    for vmnic in self.desired_state:
        if vmnic not in pnic_device:
            return False
    return True