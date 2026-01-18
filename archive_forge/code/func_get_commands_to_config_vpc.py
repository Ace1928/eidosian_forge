from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.nxos import (
def get_commands_to_config_vpc(module, vpc, domain, existing):
    vpc = dict(vpc)
    domain_only = vpc.get('domain')
    commands = []
    if 'pkl_dest' in vpc:
        pkl_command = 'peer-keepalive destination {pkl_dest}'.format(**vpc)
        if 'pkl_src' in vpc:
            pkl_command += ' source {pkl_src}'.format(**vpc)
        if 'pkl_vrf' in vpc:
            pkl_command += ' vrf {pkl_vrf}'.format(**vpc)
        commands.append(pkl_command)
    if 'auto_recovery' in vpc:
        if not vpc.get('auto_recovery'):
            vpc['auto_recovery'] = 'no'
        else:
            vpc['auto_recovery'] = ''
    if 'peer_gw' in vpc:
        if not vpc.get('peer_gw'):
            vpc['peer_gw'] = 'no'
        else:
            vpc['peer_gw'] = ''
    for param in vpc:
        command = CONFIG_ARGS.get(param)
        if command is not None:
            command = command.format(**vpc).strip()
            if 'peer-gateway' in command:
                commands.append('terminal dont-ask')
            commands.append(command)
    if commands or domain_only:
        commands.insert(0, 'vpc domain {0}'.format(domain))
    return commands