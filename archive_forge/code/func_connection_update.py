from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_text
import re
def connection_update(self, nmcli_command):
    if nmcli_command == 'create':
        cmd = [self.nmcli_bin, 'con', 'add', 'type']
        if self.tunnel_conn_type:
            cmd.append('ip-tunnel')
        else:
            cmd.append(self.type)
        cmd.append('con-name')
    elif nmcli_command == 'modify':
        cmd = [self.nmcli_bin, 'con', 'modify']
    else:
        self.module.fail_json(msg='Invalid nmcli command.')
    cmd.append(self.conn_name)
    if nmcli_command == 'create' and self.ifname is None:
        ifname = self.conn_name
    else:
        ifname = self.ifname
    options = {'connection.interface-name': ifname}
    if self.type == 'vpn' and self.ifname is None:
        del options['connection.interface-name']
    options.update(self.connection_options())
    for key, value in options.items():
        if value is not None:
            if key in self.SECRET_OPTIONS:
                self.edit_commands += ['set %s %s' % (key, value)]
                continue
            if key == 'xmit_hash_policy':
                cmd.extend(['+bond.options', 'xmit_hash_policy=%s' % value])
                continue
            cmd.extend([key, value])
    return self.execute_command(cmd)