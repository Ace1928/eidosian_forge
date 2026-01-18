from __future__ import absolute_import, division, print_function
import time
from ansible.module_utils.basic import AnsibleModule
def set_VM_Host(self, vmname, vmhost):
    VM = self.get_VM(vmname)
    HOST = self.get_Host(vmhost)
    try:
        VM.placement_policy.host = HOST
        VM.update()
        setMsg('Set startup host to ' + vmhost)
        setChanged()
    except Exception as e:
        setMsg('Failed to set startup host.')
        setMsg(str(e))
        setFailed()
        return False
    return True