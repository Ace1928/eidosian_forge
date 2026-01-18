from __future__ import absolute_import, division, print_function
from copy import deepcopy
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.argspec.vrfs.vrfs import VrfsArgs
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible.module_utils.connection import ConnectionError
def get_vrf_interfaces_from_network_instances(self, network_instances):
    vrf_interfaces = []
    for each_ins in network_instances:
        vrf_interface = dict()
        name = each_ins['name']
        if name.startswith('Vrf') or name == 'mgmt':
            vrf_interface['name'] = name
            if each_ins.get('interfaces'):
                interfaces = [{'name': intf.get('id')} for intf in each_ins['interfaces']['interface']]
                vrf_interface['members'] = {'interfaces': interfaces}
            vrf_interfaces.append(vrf_interface)
    return vrf_interfaces