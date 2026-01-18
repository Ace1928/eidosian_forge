from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.nxos import (
def get_aaa_server_info(server_type, module):
    aaa_server_info = {}
    server_command = 'show {0}-server'.format(server_type)
    request_command = 'show {0}-server directed-request'.format(server_type)
    global_key_command = 'show run | sec {0}'.format(server_type)
    aaa_regex = '.*{0}-server\\skey\\s\\d\\s+(?P<key>\\S+).*'.format(server_type)
    server_body = execute_show_command(server_command, module)[0]
    split_server = server_body.splitlines()
    for line in split_server:
        if line.startswith('timeout'):
            aaa_server_info['server_timeout'] = line.split(':')[1]
        elif line.startswith('deadtime'):
            aaa_server_info['deadtime'] = line.split(':')[1]
    request_body = execute_show_command(request_command, module)[0]
    if bool(request_body):
        aaa_server_info['directed_request'] = request_body.replace('\n', '')
    else:
        aaa_server_info['directed_request'] = 'disabled'
    key_body = execute_show_command(global_key_command, module)[0]
    try:
        match_global_key = re.match(aaa_regex, key_body, re.DOTALL)
        group_key = match_global_key.groupdict()
        aaa_server_info['global_key'] = group_key['key'].replace('"', '')
    except (AttributeError, TypeError):
        aaa_server_info['global_key'] = None
    return aaa_server_info