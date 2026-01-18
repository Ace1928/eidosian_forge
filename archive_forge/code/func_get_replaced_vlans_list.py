from __future__ import (absolute_import, division, print_function)
import re
from copy import deepcopy
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts import Facts
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible.module_utils.connection import ConnectionError
def get_replaced_vlans_list(self, want_data, have_data, protocol):
    vlans_list = []
    requests = []
    for vlan in want_data:
        vlan_id = vlan.get('vlan_id', None)
        hello_time = vlan.get('hello_time', None)
        max_age = vlan.get('max_age', None)
        fwd_delay = vlan.get('fwd_delay', None)
        bridge_priority = vlan.get('bridge_priority', None)
        interfaces = vlan.get('interfaces', None)
        for cfg_vlan in have_data:
            cfg_vlan_id = cfg_vlan.get('vlan_id', None)
            cfg_hello_time = cfg_vlan.get('hello_time', None)
            cfg_max_age = cfg_vlan.get('max_age', None)
            cfg_fwd_delay = cfg_vlan.get('fwd_delay', None)
            cfg_bridge_priority = cfg_vlan.get('bridge_priority', None)
            cfg_interfaces = cfg_vlan.get('interfaces', None)
            if vlan_id == cfg_vlan_id:
                if hello_time and hello_time != cfg_hello_time or (max_age and max_age != cfg_max_age) or (fwd_delay and fwd_delay != cfg_fwd_delay) or (bridge_priority and bridge_priority != cfg_bridge_priority):
                    vlans_list.append(cfg_vlan)
                    if cfg_hello_time:
                        if protocol == 'pvst':
                            requests.append(self.get_delete_pvst_vlan_cfg_attr(cfg_vlan_id, 'hello-time'))
                        elif protocol == 'rapid_pvst':
                            requests.append(self.get_delete_rapid_pvst_vlan_cfg_attr(cfg_vlan_id, 'hello-time'))
                    if cfg_max_age:
                        if protocol == 'pvst':
                            requests.append(self.get_delete_pvst_vlan_cfg_attr(cfg_vlan_id, 'max-age'))
                        elif protocol == 'rapid_pvst':
                            requests.append(self.get_delete_rapid_pvst_vlan_cfg_attr(cfg_vlan_id, 'max-age'))
                    if cfg_fwd_delay:
                        if protocol == 'pvst':
                            requests.append(self.get_delete_pvst_vlan_cfg_attr(cfg_vlan_id, 'forwarding-delay'))
                        elif protocol == 'rapid_pvst':
                            requests.append(self.get_delete_rapid_pvst_vlan_cfg_attr(cfg_vlan_id, 'forwarding-delay'))
                    if cfg_bridge_priority:
                        if protocol == 'pvst':
                            requests.append(self.get_delete_pvst_vlan_cfg_attr(cfg_vlan_id, 'bridge-priority'))
                        elif protocol == 'rapid_pvst':
                            requests.append(self.get_delete_rapid_pvst_vlan_cfg_attr(cfg_vlan_id, 'bridge-priority'))
                    if cfg_interfaces:
                        for cfg_intf in cfg_interfaces:
                            cfg_intf_name = cfg_intf.get('intf_name', None)
                            if protocol == 'pvst':
                                requests.append(self.get_delete_pvst_intf(cfg_vlan_id, cfg_intf_name))
                            elif protocol == 'rapid_pvst':
                                requests.append(self.get_delete_rapid_pvst_intf(cfg_vlan_id, cfg_intf_name))
                elif interfaces and cfg_interfaces:
                    intf_list = []
                    for intf in interfaces:
                        intf_name = intf.get('intf_name', None)
                        for cfg_intf in cfg_interfaces:
                            cfg_intf_name = cfg_intf.get('intf_name', None)
                            if intf_name == cfg_intf_name:
                                if intf != cfg_intf:
                                    intf_list.append(cfg_intf)
                                    vlans_list.append({'vlan_id': cfg_vlan_id, 'interfaces': intf_list})
                                    if protocol == 'pvst':
                                        requests.append(self.get_delete_pvst_intf(cfg_vlan_id, cfg_intf_name))
                                    elif protocol == 'rapid_pvst':
                                        requests.append(self.get_delete_rapid_pvst_intf(cfg_vlan_id, cfg_intf_name))
    return (vlans_list, requests)