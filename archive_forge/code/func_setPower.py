from __future__ import absolute_import, division, print_function
import time
from ansible.module_utils.basic import AnsibleModule
def setPower(self, vmname, state, timeout):
    self.__get_conn()
    VM = self.conn.get_VM(vmname)
    if VM is None:
        setMsg('VM does not exist.')
        setFailed()
        return False
    if state == VM.status.state:
        setMsg('VM state was already ' + state)
    else:
        if state == 'up':
            setMsg('VM is going to start')
            self.conn.start_VM(vmname, timeout)
            setChanged()
        elif state == 'down':
            setMsg('VM is going to stop')
            self.conn.stop_VM(vmname, timeout)
            setChanged()
        elif state == 'restarted':
            self.setPower(vmname, 'down', timeout)
            checkFail()
            self.setPower(vmname, 'up', timeout)
        checkFail()
        setMsg('the vm state is set to ' + state)
    return True