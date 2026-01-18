from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.network import is_mac
def delete_nictag(self):
    cmd = [self.nictagadm_bin, '-v', 'delete']
    if self.force:
        cmd.append('-f')
    cmd.append(self.name)
    return self.module.run_command(cmd)