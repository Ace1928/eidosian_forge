from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.nxos import (
def processShowVsanMembership(self):
    patv = '^vsan\\s+(\\d+).*'
    output = self.execute_show_vsan_mem_cmd().split('\n')
    memlist = []
    v = None
    for o in output:
        z = re.match(patv, o.strip())
        if z:
            if v is not None:
                self.vsaninfo[v].vsaninterfaces = memlist
                memlist = []
            v = z.group(1)
        if 'interfaces' not in o:
            llist = o.strip().split()
            memlist = memlist + llist
    self.vsaninfo[v].vsaninterfaces = memlist