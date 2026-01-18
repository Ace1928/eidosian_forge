from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
def delete_vnic(self):
    cmd = [self.module.get_bin_path('dladm', True)]
    cmd.append('delete-vnic')
    if self.temporary:
        cmd.append('-t')
    cmd.append(self.name)
    return self.module.run_command(cmd)