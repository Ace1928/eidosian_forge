from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.nxos import (
def get_vtp_password(module):
    command = 'show vtp password'
    output = 'json'
    cap = get_capabilities(module)['device_info']['network_os_model']
    if re.search('Nexus 6', cap):
        output = 'text'
    body = execute_show_command(command, module, output)[0]
    if output == 'json':
        password = body.get('passwd', '')
    else:
        password = ''
        rp = 'VTP Password: (\\S+)'
        mo = re.search(rp, body)
        if mo:
            password = mo.group(1)
    return str(password)