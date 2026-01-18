from __future__ import absolute_import, division, print_function
import time
from ansible.module_utils.basic import AnsibleModule
def get_VM(self, vm_name):
    return self.conn.vms.get(name=vm_name)