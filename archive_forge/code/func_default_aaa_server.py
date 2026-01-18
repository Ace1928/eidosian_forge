from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.nxos import (
def default_aaa_server(existing, params, server_type):
    cmds = []
    deadtime = params.get('deadtime')
    server_timeout = params.get('server_timeout')
    directed_request = params.get('directed_request')
    global_key = params.get('global_key')
    existing_key = existing.get('global_key')
    if deadtime is not None and existing.get('deadtime') != PARAM_TO_DEFAULT_KEYMAP['deadtime']:
        cmds.append('no {0}-server deadtime 1'.format(server_type))
    if server_timeout is not None and existing.get('server_timeout') != PARAM_TO_DEFAULT_KEYMAP['server_timeout']:
        cmds.append('no {0}-server timeout 1'.format(server_type))
    if directed_request is not None and existing.get('directed_request') != PARAM_TO_DEFAULT_KEYMAP['directed_request']:
        cmds.append('no {0}-server directed-request'.format(server_type))
    if global_key is not None and existing_key is not None:
        cmds.append('no {0}-server key 7 {1}'.format(server_type, existing_key))
    return cmds