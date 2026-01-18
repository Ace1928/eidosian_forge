from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.nxos import (
def isZonePresentInZoneset(self, zsname, zname):
    if zsname in self.zsDetails.keys():
        return zname in self.zsDetails[zsname]
    return False