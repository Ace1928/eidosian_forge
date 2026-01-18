from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.nxos import (
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.utils.utils import Version
def parse_https(data):
    https_res = ['nxapi https port (\\d+)']
    https_port = None
    for regex in https_res:
        match = re.search(regex, data, re.M)
        if match:
            https_port = int(match.group(1))
            break
    return {'https': https_port is not None, 'https_port': https_port}