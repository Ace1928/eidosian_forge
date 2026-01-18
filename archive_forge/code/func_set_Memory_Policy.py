from __future__ import absolute_import, division, print_function
import time
from ansible.module_utils.basic import AnsibleModule
def set_Memory_Policy(self, name, memory_policy):
    VM = self.get_VM(name)
    VM.memory_policy.guaranteed = int(memory_policy) * 1024 * 1024 * 1024
    try:
        VM.update()
        setMsg('The memory policy has been updated.')
        setChanged()
        return True
    except Exception as e:
        setMsg('Failed to update memory policy.')
        setMsg(str(e))
        setFailed()
        return False