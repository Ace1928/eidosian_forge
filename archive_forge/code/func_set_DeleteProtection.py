from __future__ import absolute_import, division, print_function
import time
from ansible.module_utils.basic import AnsibleModule
def set_DeleteProtection(self, vmname, del_prot):
    VM = self.get_VM(vmname)
    VM.delete_protected = del_prot
    try:
        VM.update()
        setChanged()
    except Exception as e:
        setMsg('Failed to update delete protection.')
        setMsg(str(e))
        setFailed()
        return False
    return True