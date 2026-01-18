from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
def list_vms(self, state=None):
    self.conn = self.__get_conn()
    vms = self.conn.find_vm(-1)
    results = []
    for x in vms:
        try:
            if state:
                vmstate = self.conn.get_status2(x)
                if vmstate == state:
                    results.append(x.name())
            else:
                results.append(x.name())
        except Exception:
            pass
    return results