from __future__ import (absolute_import, division, print_function)
import re
from copy import deepcopy
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts import Facts
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible.module_utils.connection import ConnectionError
def get_delete_stp_rapid_pvst_requests(self, commands, have):
    requests = []
    rapid_pvst = commands.get('rapid_pvst', None)
    if rapid_pvst:
        vlans_list = []
        for vlan in rapid_pvst:
            vlans_dict = {}
            vlan_id = vlan.get('vlan_id', None)
            hello_time = vlan.get('hello_time', None)
            max_age = vlan.get('max_age', None)
            fwd_delay = vlan.get('fwd_delay', None)
            bridge_priority = vlan.get('bridge_priority', None)
            interfaces = vlan.get('interfaces', [])
            cfg_rapid_pvst = have.get('rapid_pvst', None)
            if cfg_rapid_pvst:
                for cfg_vlan in cfg_rapid_pvst:
                    cfg_vlan_id = cfg_vlan.get('vlan_id', None)
                    cfg_hello_time = cfg_vlan.get('hello_time', None)
                    cfg_max_age = cfg_vlan.get('max_age', None)
                    cfg_fwd_delay = cfg_vlan.get('fwd_delay', None)
                    cfg_bridge_priority = cfg_vlan.get('bridge_priority', None)
                    cfg_interfaces = cfg_vlan.get('interfaces', [])
                    if vlan_id == cfg_vlan_id:
                        if hello_time and hello_time == cfg_hello_time:
                            requests.append(self.get_delete_rapid_pvst_vlan_cfg_attr(vlan_id, 'hello-time'))
                            vlans_dict.update({'vlan_id': vlan_id, 'hello_time': hello_time})
                        if max_age and max_age == cfg_max_age:
                            requests.append(self.get_delete_rapid_pvst_vlan_cfg_attr(vlan_id, 'max-age'))
                            vlans_dict.update({'vlan_id': vlan_id, 'max_age': max_age})
                        if fwd_delay and fwd_delay == cfg_fwd_delay:
                            requests.append(self.get_delete_rapid_pvst_vlan_cfg_attr(vlan_id, 'forwarding-delay'))
                            vlans_dict.update({'vlan_id': vlan_id, 'fwd_delay': fwd_delay})
                        if bridge_priority and bridge_priority == cfg_bridge_priority:
                            requests.append(self.get_delete_rapid_pvst_vlan_cfg_attr(vlan_id, 'bridge-priority'))
                            vlans_dict.update({'vlan_id': vlan_id, 'bridge_priority': bridge_priority})
                        if interfaces:
                            intf_list = []
                            for intf in interfaces:
                                intf_dict = {}
                                intf_name = intf.get('intf_name', None)
                                cost = intf.get('cost', None)
                                port_priority = intf.get('port_priority', None)
                                if cfg_interfaces:
                                    for cfg_intf in cfg_interfaces:
                                        cfg_intf_name = cfg_intf.get('intf_name', None)
                                        cfg_cost = cfg_intf.get('cost', None)
                                        cfg_port_priority = cfg_intf.get('port_priority', None)
                                        if intf_name == cfg_intf_name:
                                            if cost and cost == cfg_cost:
                                                requests.append(self.get_delete_rapid_pvst_intf_cfg_attr(vlan_id, intf_name, 'cost'))
                                                intf_dict.update({'intf_name': intf_name, 'cost': cost})
                                            if port_priority and port_priority == cfg_port_priority:
                                                requests.append(self.get_delete_rapid_pvst_intf_cfg_attr(vlan_id, intf_name, 'port-priority'))
                                                intf_dict.update({'intf_name': intf_name, 'port_priority': port_priority})
                                            if not cost and (not port_priority):
                                                requests.append(self.get_delete_rapid_pvst_intf(vlan_id, intf_name))
                                                intf_dict.update({'intf_name': intf_name})
                                            if intf_dict:
                                                intf_list.append(intf_dict)
                            if intf_list:
                                vlans_dict.update({'vlan_id': vlan_id, 'interfaces': intf_list})
                        if vlans_dict:
                            vlans_list.append(vlans_dict)
        if vlans_list:
            commands['rapid_pvst'] = vlans_list
        else:
            commands.pop('rapid_pvst')
    return requests