from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
def reset_property(self):
    cmd = [self.dladm_bin]
    cmd.append('reset-linkprop')
    if self.temporary:
        cmd.append('-t')
    cmd.append('-p')
    cmd.append(self.property)
    cmd.append(self.link)
    return self.module.run_command(cmd)