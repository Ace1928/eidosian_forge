from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.community.vmware.plugins.module_utils.vmware import (
def check_dvs(self):
    """Check if DVS is present"""
    self.dvs = find_dvs_by_name(self.content, self.switch_name, folder=self.folder_obj)
    if self.dvs is None:
        return 'absent'
    return 'present'