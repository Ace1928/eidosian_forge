from __future__ import absolute_import, division, print_function
import time
from ansible.module_utils.basic import AnsibleModule
def remove_VM(self, vmname):
    VM = self.get_VM(vmname)
    try:
        VM.delete()
    except Exception as e:
        setMsg('Failed to remove VM.')
        setMsg(str(e))
        setFailed()
        return False
    return True