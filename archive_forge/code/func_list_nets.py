from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
def list_nets(self, state=None):
    results = []
    for entry in self.conn.find_entry(-1):
        if state:
            if state == self.conn.get_status2(entry):
                results.append(entry.name())
        else:
            results.append(entry.name())
    return results