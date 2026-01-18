from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.nxos import (
def get_existing_portchannel_to_vpc_mappings(module):
    pc_vpc_mapping = {}
    try:
        body = run_commands(module, ['show vpc brief | json'])[0]
        vpc_table = body['TABLE_vpc']['ROW_vpc']
    except (KeyError, AttributeError, TypeError):
        vpc_table = None
    if vpc_table:
        if isinstance(vpc_table, dict):
            vpc_table = [vpc_table]
        for vpc in vpc_table:
            pc_vpc_mapping[str(vpc['vpc-id'])] = str(vpc['vpc-ifindex'])
    return pc_vpc_mapping