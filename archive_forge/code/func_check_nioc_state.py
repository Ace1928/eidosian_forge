from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.community.vmware.plugins.module_utils.vmware import (
def check_nioc_state(self):
    self.dvs = find_dvs_by_name(self.content, self.switch)
    if self.dvs is None:
        self.module.fail_json(msg='DVS %s was not found.' % self.switch)
    else:
        if not self.dvs.config.networkResourceManagementEnabled:
            return 'absent'
        if self.version and self.dvs.config.networkResourceControlVersion != self.version:
            return 'version'
        return self.check_resources()