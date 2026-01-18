from __future__ import absolute_import, division, print_function
import time
from ansible.module_utils.basic import AnsibleModule
def del_NIC(self, vmname, nicname):
    return self.get_NIC(vmname, nicname).delete()