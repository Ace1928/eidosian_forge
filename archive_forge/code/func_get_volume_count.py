from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
def get_volume_count(self, entryid):
    return self.find_entry(entryid).numOfVolumes()