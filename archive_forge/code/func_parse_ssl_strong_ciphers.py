from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.nxos import (
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.utils.utils import Version
def parse_ssl_strong_ciphers(data):
    ciphers_res = ['(\\w+) nxapi ssl ciphers weak']
    value = None
    for regex in ciphers_res:
        match = re.search(regex, data, re.M)
        if match:
            value = match.group(1)
            break
    return {'ssl_strong_ciphers': value == 'no'}