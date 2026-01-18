from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import (
def state_update_dvspg(self):
    changed = True
    result = None
    if not self.module.check_mode:
        changed, result = self.update_port_group()
    self.module.exit_json(changed=changed, result=str(result))