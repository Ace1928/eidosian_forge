from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
def get_maxVcpus(self, vmid):
    """
        Gets the max number of VCPUs on a guest
        """
    self.__get_conn()
    return self.conn.get_maxVcpus(vmid)