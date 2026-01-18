from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.nxos import (
def get_portchannel_vpc_config(module, portchannel):
    peer_link_pc = None
    peer_link = False
    vpc = ''
    pc = ''
    config = {}
    try:
        body = run_commands(module, ['show vpc brief | json'])[0]
        table = body['TABLE_peerlink']['ROW_peerlink']
    except (KeyError, AttributeError, TypeError):
        table = {}
    if table:
        peer_link_pc = table.get('peerlink-ifindex', None)
    if peer_link_pc:
        plpc = str(peer_link_pc[2:])
        if portchannel == plpc:
            config['portchannel'] = portchannel
            config['peer-link'] = True
            config['vpc'] = vpc
    mapping = get_existing_portchannel_to_vpc_mappings(module)
    for existing_vpc, port_channel in mapping.items():
        port_ch = str(port_channel[2:])
        if port_ch == portchannel:
            pc = port_ch
            vpc = str(existing_vpc)
            config['portchannel'] = pc
            config['peer-link'] = peer_link
            config['vpc'] = vpc
    return config