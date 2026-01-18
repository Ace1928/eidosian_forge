from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.nxos import (
def isZoneMemberPresent(self, zname, cmd):
    if zname in self.zDetails.keys():
        zonememlist = self.zDetails[zname]
        for eachline in zonememlist:
            if cmd in eachline:
                return True
    return False