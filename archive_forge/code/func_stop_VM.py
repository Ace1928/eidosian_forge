from __future__ import absolute_import, division, print_function
import time
from ansible.module_utils.basic import AnsibleModule
def stop_VM(self, vmname, timeout):
    VM = self.get_VM(vmname)
    try:
        VM.stop()
    except Exception as e:
        setMsg('Failed to stop VM.')
        setMsg(str(e))
        setFailed()
        return False
    return self.wait_VM(vmname, 'down', timeout)