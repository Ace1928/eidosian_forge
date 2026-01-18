from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
def property_exists(self):
    cmd = [self.dladm_bin]
    cmd.append('show-linkprop')
    cmd.append('-p')
    cmd.append(self.property)
    cmd.append(self.link)
    rc, dummy, dummy = self.module.run_command(cmd)
    if rc == 0:
        return True
    else:
        self.module.fail_json(msg='Unknown property "%s" on link %s' % (self.property, self.link), property=self.property, link=self.link)