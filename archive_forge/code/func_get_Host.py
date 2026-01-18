from __future__ import absolute_import, division, print_function
import time
from ansible.module_utils.basic import AnsibleModule
def get_Host(self, host_name):
    return self.conn.hosts.get(name=host_name)