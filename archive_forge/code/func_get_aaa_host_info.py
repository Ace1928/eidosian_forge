from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.nxos import (
def get_aaa_host_info(module, server_type, address):
    aaa_host_info = {}
    command = 'show run | inc {0}-server.host.{1}'.format(server_type, address)
    body = execute_show_command(command, module)[0]
    if body:
        try:
            if 'radius' in body:
                pattern = '\\S+ host \\S+(?:\\s+key 7\\s+(\\S+))?(?:\\s+auth-port (\\d+))?(?:\\s+acct-port (\\d+))?(?:\\s+authentication)?(?:\\s+accounting)?(?:\\s+timeout (\\d+))?'
                match = re.search(pattern, body)
                aaa_host_info['key'] = match.group(1)
                if aaa_host_info['key']:
                    aaa_host_info['key'] = aaa_host_info['key'].replace('"', '')
                    aaa_host_info['encrypt_type'] = '7'
                aaa_host_info['auth_port'] = match.group(2)
                aaa_host_info['acct_port'] = match.group(3)
                aaa_host_info['host_timeout'] = match.group(4)
            elif 'tacacs' in body:
                pattern = '\\S+ host \\S+(?:\\s+key 7\\s+(\\S+))?(?:\\s+port (\\d+))?(?:\\s+timeout (\\d+))?'
                match = re.search(pattern, body)
                aaa_host_info['key'] = match.group(1)
                if aaa_host_info['key']:
                    aaa_host_info['key'] = aaa_host_info['key'].replace('"', '')
                    aaa_host_info['encrypt_type'] = '7'
                aaa_host_info['tacacs_port'] = match.group(2)
                aaa_host_info['host_timeout'] = match.group(3)
            aaa_host_info['server_type'] = server_type
            aaa_host_info['address'] = address
        except TypeError:
            return {}
    else:
        return {}
    return aaa_host_info