from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
def create_vnic(self):
    cmd = [self.module.get_bin_path('dladm', True)]
    cmd.append('create-vnic')
    if self.temporary:
        cmd.append('-t')
    if self.mac:
        cmd.append('-m')
        cmd.append(self.mac)
    if self.vlan:
        cmd.append('-v')
        cmd.append(self.vlan)
    cmd.append('-l')
    cmd.append(self.link)
    cmd.append(self.name)
    return self.module.run_command(cmd)