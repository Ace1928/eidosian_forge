from __future__ import absolute_import, division, print_function
import os
from ansible.module_utils.basic import AnsibleModule
def check_openzfs(self):
    cmd = [self.zpool_cmd]
    cmd.extend(['get', 'version'])
    cmd.append(self.pool)
    rc, out, err = self.module.run_command(cmd, check_rc=True)
    version = out.splitlines()[-1].split()[2]
    if version == '-':
        return True
    if int(version) == 5000:
        return True
    return False