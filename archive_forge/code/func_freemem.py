from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
def freemem(self):
    self.conn = self.__get_conn()
    return self.conn.getFreeMemory()