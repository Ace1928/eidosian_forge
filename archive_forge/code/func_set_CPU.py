from __future__ import absolute_import, division, print_function
import time
from ansible.module_utils.basic import AnsibleModule
def set_CPU(self, name, cpu):
    VM = self.get_VM(name)
    VM.cpu.topology.cores = int(cpu)
    try:
        VM.update()
        setMsg('The number of CPUs has been updated.')
        setChanged()
        return True
    except Exception as e:
        setMsg('Failed to update the number of CPUs.')
        setMsg(str(e))
        setFailed()
        return False