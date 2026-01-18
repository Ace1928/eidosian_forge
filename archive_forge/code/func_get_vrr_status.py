from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.nxos import (
def get_vrr_status(group, module, interface):
    command = 'show run all | section interface.{0}$'.format(interface)
    body = execute_show_command(command, module)
    vrf_index = None
    admin_state = 'shutdown'
    if body:
        splitted_body = body.splitlines()
        for index in range(0, len(splitted_body) - 1):
            if splitted_body[index].strip() == 'vrrp {0}'.format(group):
                vrf_index = index
        vrf_section = splitted_body[vrf_index:]
        for line in vrf_section:
            if line.strip() == 'no shutdown':
                admin_state = 'no shutdown'
                break
    return admin_state