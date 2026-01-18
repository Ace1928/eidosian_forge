from __future__ import absolute_import, division, print_function
import copy
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.nxos import (
def outliers(haves, wants):
    wants = list(wants)
    return [absent(h) for h in haves if not (h in wants or wants.append(h))]