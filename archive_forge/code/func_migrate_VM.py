from __future__ import absolute_import, division, print_function
import time
from ansible.module_utils.basic import AnsibleModule
def migrate_VM(self, vmname, vmhost):
    VM = self.get_VM(vmname)
    HOST = self.get_Host_byid(VM.host.id)
    if str(HOST.name) != vmhost:
        try:
            VM.migrate(action=params.Action(host=params.Host(name=vmhost)))
            setChanged()
            setMsg('VM migrated to ' + vmhost)
        except Exception as e:
            setMsg('Failed to set startup host.')
            setMsg(str(e))
            setFailed()
            return False
    return True