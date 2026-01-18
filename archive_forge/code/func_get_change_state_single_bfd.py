from __future__ import (absolute_import, division, print_function)
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import string_types
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config
def get_change_state_single_bfd(self):
    """get ipv4 single bfd change state"""
    self.get_single_bfd(self.state)
    change = False
    version = self.version
    if self.state == 'present':
        if self.static_routes_info['sroute_single_bfd']:
            for static_route in self.static_routes_info['sroute_single_bfd']:
                if static_route is not None and static_route['afType'] == version:
                    if self.nhp_interface:
                        if static_route['ifName'].lower() != self.nhp_interface.lower():
                            change = True
                    if self.destvrf:
                        if static_route['destVrfName'].lower() != self.destvrf.lower():
                            change = True
                    if self.next_hop:
                        if static_route['nexthop'].lower() != self.next_hop.lower():
                            change = True
                    if self.prefix:
                        if static_route['localAddress'].lower() != self.prefix.lower():
                            change = True
                    if self.min_tx_interval:
                        if int(static_route['minTxInterval']) != self.min_tx_interval:
                            change = True
                    if self.min_rx_interval:
                        if int(static_route['minRxInterval']) != self.min_rx_interval:
                            change = True
                    if self.detect_multiplier:
                        if int(static_route['multiplier']) != self.detect_multiplier:
                            change = True
                    return change
                else:
                    continue
        else:
            change = True
    else:
        for static_route in self.static_routes_info['sroute_single_bfd']:
            if static_route['ifName'] and self.nhp_interface:
                if static_route['ifName'].lower() == self.nhp_interface.lower() and static_route['nexthop'].lower() == self.next_hop.lower() and (static_route['afType'] == version):
                    change = True
                    return change
            if static_route['destVrfName'] and self.destvrf:
                if static_route['destVrfName'].lower() == self.destvrf.lower() and static_route['nexthop'].lower() == self.next_hop.lower() and (static_route['afType'] == version):
                    change = True
                    return change
            if static_route['nexthop'] and self.next_hop:
                if static_route['nexthop'].lower() == self.next_hop.lower() and static_route['afType'] == version:
                    change = True
                    return change
            else:
                continue
        change = False
    return change