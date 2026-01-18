from __future__ import absolute_import, division, print_function
import time
from ansible.module_utils.basic import AnsibleModule
def createVMimage(self, name, cluster, template, disks):
    self.__get_conn()
    return self.conn.createVMimage(name, cluster, template, disks)