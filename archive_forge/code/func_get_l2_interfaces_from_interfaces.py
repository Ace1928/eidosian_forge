from __future__ import absolute_import, division, print_function
from copy import deepcopy
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.argspec.l2_interfaces.l2_interfaces import L2_interfacesArgs
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible.module_utils.connection import ConnectionError
def get_l2_interfaces_from_interfaces(self, interfaces):
    l2_interfaces = []
    for intf in interfaces:
        name = intf['name']
        key = 'openconfig-if-ethernet:ethernet'
        if name.startswith('PortChannel'):
            key = 'openconfig-if-aggregate:aggregation'
        eth_det = intf.get(key)
        if eth_det:
            open_cfg_vlan = eth_det.get('openconfig-vlan:switched-vlan')
            if open_cfg_vlan:
                new_det = dict()
                new_det['name'] = name
                if name == 'eth0':
                    continue
                if open_cfg_vlan['config'].get('access-vlan'):
                    new_det['access'] = dict({'vlan': open_cfg_vlan['config'].get('access-vlan')})
                if open_cfg_vlan['config'].get('trunk-vlans'):
                    new_det['trunk'] = {}
                    new_det['trunk']['allowed_vlans'] = []
                    for vlan in open_cfg_vlan['config'].get('trunk-vlans'):
                        vlan_argspec = ''
                        if isinstance(vlan, str):
                            vlan_argspec = vlan.replace('"', '')
                            if '..' in vlan_argspec:
                                vlan_argspec = vlan_argspec.replace('..', '-')
                        else:
                            vlan_argspec = str(vlan)
                        new_det['trunk']['allowed_vlans'].append({'vlan': vlan_argspec})
                l2_interfaces.append(new_det)
    return l2_interfaces