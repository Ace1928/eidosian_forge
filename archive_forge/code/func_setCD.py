from __future__ import absolute_import, division, print_function
import time
from ansible.module_utils.basic import AnsibleModule
def setCD(self, vmname, cd_drive):
    self.__get_conn()
    if cd_drive:
        return self.conn.set_CD(vmname, cd_drive)
    else:
        return self.conn.remove_CD(vmname)