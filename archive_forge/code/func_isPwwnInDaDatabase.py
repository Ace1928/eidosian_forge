from __future__ import absolute_import, division, print_function
import string
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.nxos import (
def isPwwnInDaDatabase(self, pwwn):
    newpwwn = ':'.join(['0' + str(ep) if len(ep) == 1 else ep for ep in pwwn.split(':')])
    return newpwwn in self.da_dict.values()