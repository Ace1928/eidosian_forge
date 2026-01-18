from __future__ import absolute_import, division, print_function
import string
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.nxos import (
def getPwwnByName(self, name):
    if name in self.da_dict.keys():
        return self.da_dict[name]
    else:
        return None