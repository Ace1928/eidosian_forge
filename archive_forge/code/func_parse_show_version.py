from __future__ import absolute_import, division, print_function
import re
from time import sleep
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.nxos import (
def parse_show_version(data):
    version_data = {'raw': data[0].split('\n')}
    version_data['version'] = ''
    version_data['error'] = False
    for x in version_data['raw']:
        mo = re.search('(kickstart|system|NXOS):\\s+version\\s+(\\S+)', x)
        if mo:
            version_data['version'] = mo.group(2)
            continue
    if version_data['version'] == '':
        version_data['error'] = True
    return version_data