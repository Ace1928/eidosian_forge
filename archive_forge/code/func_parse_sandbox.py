from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.nxos import (
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.utils.utils import Version
def parse_sandbox(data):
    sandbox = [item for item in data.split('\n') if re.search('.*sandbox.*', item)]
    value = False
    if sandbox and sandbox[0] == 'nxapi sandbox':
        value = True
    return {'sandbox': value}