from __future__ import absolute_import, division, print_function
import time
from ansible.module_utils.basic import AnsibleModule
def setDeleteProtection(self, vmname, del_prot):
    self.__get_conn()
    VM = self.conn.get_VM(vmname)
    if bool(VM.delete_protected) != bool(del_prot):
        self.conn.set_DeleteProtection(vmname, del_prot)
        checkFail()
        setMsg('`delete protection` has been updated.')
    else:
        setMsg('`delete protection` already has the right value.')
    return True