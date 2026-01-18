from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.nxos import (
def auth_type_to_num(auth_type):
    if auth_type == 'encrypt':
        return '7'
    else:
        return '0'