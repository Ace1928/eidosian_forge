from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.nxos import (
def config_aaa_server(params, server_type):
    cmds = []
    deadtime = params.get('deadtime')
    server_timeout = params.get('server_timeout')
    directed_request = params.get('directed_request')
    encrypt_type = params.get('encrypt_type', '7')
    global_key = params.get('global_key')
    if deadtime is not None:
        cmds.append('{0}-server deadtime {1}'.format(server_type, deadtime))
    if server_timeout is not None:
        cmds.append('{0}-server timeout {1}'.format(server_type, server_timeout))
    if directed_request is not None:
        if directed_request == 'enabled':
            cmds.append('{0}-server directed-request'.format(server_type))
        elif directed_request == 'disabled':
            cmds.append('no {0}-server directed-request'.format(server_type))
    if global_key is not None:
        cmds.append('{0}-server key {1} {2}'.format(server_type, encrypt_type, global_key))
    return cmds