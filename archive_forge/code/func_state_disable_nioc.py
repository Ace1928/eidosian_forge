from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.community.vmware.plugins.module_utils.vmware import (
def state_disable_nioc(self):
    self.result['changed'] = True
    if not self.module.check_mode:
        self.set_nioc_enabled(False)
    self.result['dvswitch_nioc_status'] = 'Disabled NIOC'