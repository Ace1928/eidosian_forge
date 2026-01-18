from __future__ import absolute_import, division, print_function
import time
from ansible.module_utils.basic import AnsibleModule
def setMemoryPolicy(self, name, memory_policy):
    self.__get_conn()
    return self.conn.set_Memory_Policy(name, memory_policy)