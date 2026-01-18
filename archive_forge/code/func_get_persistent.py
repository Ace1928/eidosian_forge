from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
def get_persistent(self, entryid):
    state = self.find_entry(entryid).isPersistent()
    return ENTRY_STATE_PERSISTENT_MAP.get(state, 'unknown')