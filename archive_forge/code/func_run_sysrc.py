from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
import re
def run_sysrc(self, *args):
    cmd = [self.sysrc, '-f', self.path]
    if self.jail:
        cmd += ['-j', self.jail]
    cmd.extend(args)
    rc, out, err = self.module.run_command(cmd)
    return (rc, out, err)