from __future__ import absolute_import, division, print_function
import os
from ansible.module_utils.basic import AnsibleModule
def check_enhanced_sharing(self):
    if self.is_solaris and (not self.is_openzfs):
        cmd = [self.zpool_cmd]
        cmd.extend(['get', 'version'])
        cmd.append(self.pool)
        rc, out, err = self.module.run_command(cmd, check_rc=True)
        version = out.splitlines()[-1].split()[2]
        if int(version) >= 34:
            return True
    return False