from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
def create_iptun(self):
    cmd = [self.dladm_bin]
    cmd.append('create-iptun')
    if self.temporary:
        cmd.append('-t')
    cmd.append('-T')
    cmd.append(self.type)
    cmd.append('-a')
    cmd.append('local=' + self.local_address + ',remote=' + self.remote_address)
    cmd.append(self.name)
    return self.module.run_command(cmd)