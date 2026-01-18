from __future__ import absolute_import, division, print_function
import time
from ansible.module_utils.basic import AnsibleModule
def set_CPU_share(self, name, cpu_share):
    VM = self.get_VM(name)
    VM.cpu_shares = int(cpu_share)
    try:
        VM.update()
        setMsg('The CPU share has been updated.')
        setChanged()
        return True
    except Exception as e:
        setMsg('Failed to update the CPU share.')
        setMsg(str(e))
        setFailed()
        return False