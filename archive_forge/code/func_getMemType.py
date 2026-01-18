from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.nxos import (
def getMemType(supported_choices, allmemkeys, default='pwwn'):
    for eachchoice in supported_choices:
        if eachchoice in allmemkeys:
            return eachchoice
    return default