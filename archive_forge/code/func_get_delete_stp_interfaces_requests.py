from __future__ import (absolute_import, division, print_function)
import re
from copy import deepcopy
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts import Facts
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible.module_utils.connection import ConnectionError
def get_delete_stp_interfaces_requests(self, commands, have):
    requests = []
    interfaces = commands.get('interfaces', None)
    if interfaces:
        intf_list = []
        for intf in interfaces:
            intf_dict = {}
            intf_name = intf.get('intf_name', None)
            edge_port = intf.get('edge_port', None)
            link_type = intf.get('link_type', None)
            guard = intf.get('guard', None)
            bpdu_guard = intf.get('bpdu_guard', None)
            bpdu_filter = intf.get('bpdu_filter', None)
            portfast = intf.get('portfast', None)
            uplink_fast = intf.get('uplink_fast', None)
            shutdown = intf.get('shutdown', None)
            cost = intf.get('cost', None)
            port_priority = intf.get('port_priority', None)
            stp_enable = intf.get('stp_enable', None)
            cfg_interfaces = have.get('interfaces', None)
            if cfg_interfaces:
                for cfg_intf in cfg_interfaces:
                    cfg_intf_name = cfg_intf.get('intf_name', None)
                    cfg_edge_port = cfg_intf.get('edge_port', None)
                    cfg_link_type = cfg_intf.get('link_type', None)
                    cfg_guard = cfg_intf.get('guard', None)
                    cfg_bpdu_guard = cfg_intf.get('bpdu_guard', None)
                    cfg_bpdu_filter = cfg_intf.get('bpdu_filter', None)
                    cfg_portfast = cfg_intf.get('portfast', None)
                    cfg_uplink_fast = cfg_intf.get('uplink_fast', None)
                    cfg_shutdown = cfg_intf.get('shutdown', None)
                    cfg_cost = cfg_intf.get('cost', None)
                    cfg_port_priority = cfg_intf.get('port_priority', None)
                    cfg_stp_enable = cfg_intf.get('stp_enable', None)
                    if intf_name and intf_name == cfg_intf_name:
                        if edge_port and edge_port == cfg_edge_port:
                            requests.append(self.get_delete_stp_interface_attr(intf_name, 'edge-port'))
                            intf_dict.update({'intf_name': intf_name, 'edge_port': edge_port})
                        if link_type and link_type == cfg_link_type:
                            requests.append(self.get_delete_stp_interface_attr(intf_name, 'link-type'))
                            intf_dict.update({'intf_name': intf_name, 'link_type': link_type})
                        if guard and guard == cfg_guard:
                            requests.append(self.get_delete_stp_interface_attr(intf_name, 'guard'))
                            intf_dict.update({'intf_name': intf_name, 'guard': guard})
                        if bpdu_guard and bpdu_guard == cfg_bpdu_guard:
                            url = '%s/interfaces/interface=%s/config/bpdu-guard' % (STP_PATH, intf_name)
                            payload = {'openconfig-spanning-tree:bpdu-guard': False}
                            request = {'path': url, 'method': PATCH, 'data': payload}
                            requests.append(request)
                            intf_dict.update({'intf_name': intf_name, 'bpdu_guard': bpdu_guard})
                        if bpdu_filter and bpdu_filter == cfg_bpdu_filter:
                            requests.append(self.get_delete_stp_interface_attr(intf_name, 'bpdu-filter'))
                            intf_dict.update({'intf_name': intf_name, 'bpdu_filter': bpdu_filter})
                        if portfast and portfast == cfg_portfast:
                            requests.append(self.get_delete_stp_interface_attr(intf_name, 'openconfig-spanning-tree-ext:portfast'))
                            intf_dict.update({'intf_name': intf_name, 'portfast': portfast})
                        if uplink_fast and uplink_fast == cfg_uplink_fast:
                            url = '%s/interfaces/interface=%s/config/openconfig-spanning-tree-ext:uplink-fast' % (STP_PATH, intf_name)
                            payload = {'openconfig-spanning-tree-ext:uplink-fast': False}
                            request = {'path': url, 'method': PATCH, 'data': payload}
                            requests.append(request)
                            intf_dict.update({'intf_name': intf_name, 'uplink_fast': uplink_fast})
                        if shutdown and shutdown == cfg_shutdown:
                            url = '%s/interfaces/interface=%s/config/openconfig-spanning-tree-ext:bpdu-guard-port-shutdown' % (STP_PATH, intf_name)
                            payload = {'openconfig-spanning-tree-ext:bpdu-guard-port-shutdown': False}
                            request = {'path': url, 'method': PATCH, 'data': payload}
                            requests.append(request)
                            intf_dict.update({'intf_name': intf_name, 'shutdown': shutdown})
                        if cost and cost == cfg_cost:
                            requests.append(self.get_delete_stp_interface_attr(intf_name, 'openconfig-spanning-tree-ext:cost'))
                            intf_dict.update({'intf_name': intf_name, 'cost': cost})
                        if port_priority and port_priority == cfg_port_priority:
                            requests.append(self.get_delete_stp_interface_attr(intf_name, 'openconfig-spanning-tree-ext:port-priority'))
                            intf_dict.update({'intf_name': intf_name, 'port_priority': port_priority})
                        if stp_enable is False and stp_enable == cfg_stp_enable:
                            url = '%s/interfaces/interface=%s/config/openconfig-spanning-tree-ext:spanning-tree-enable' % (STP_PATH, intf_name)
                            payload = {'openconfig-spanning-tree-ext:spanning-tree-enable': True}
                            request = {'path': url, 'method': PATCH, 'data': payload}
                            requests.append(request)
                            intf_dict.update({'intf_name': intf_name, 'stp_enable': stp_enable})
                        if edge_port is None and (not link_type) and (not guard) and (bpdu_guard is None) and (bpdu_filter is None) and (portfast is None) and (uplink_fast is None) and (shutdown is None) and (not cost) and (not port_priority) and (stp_enable is None):
                            requests.append(self.get_delete_stp_interface(intf_name))
                            intf_dict.update({'intf_name': intf_name})
                        if intf_dict:
                            intf_list.append(intf_dict)
        if intf_list:
            commands['interfaces'] = intf_list
        else:
            commands.pop('interfaces')
    return requests