from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
def get_autostart(self, vmid):
    vm = self.conn.lookupByName(vmid)
    return vm.autostart()