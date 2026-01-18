from __future__ import (absolute_import, division, print_function)
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import string_types
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config
def get_ip_static_route(self):
    """get ip static route"""
    change = False
    version = self.version
    self.get_static_route(self.state)
    change_list = list()
    if self.state == 'present':
        for static_route in self.static_routes_info['sroute']:
            if self.is_prefix_exist(static_route, self.version):
                info_dict = dict()
                exist_dict = dict()
                if self.vrf:
                    info_dict['vrfName'] = self.vrf
                    exist_dict['vrfName'] = static_route['vrfName']
                if self.destvrf:
                    info_dict['destVrfName'] = self.destvrf
                    exist_dict['destVrfName'] = static_route['destVrfName']
                if self.description:
                    info_dict['description'] = self.description
                    exist_dict['description'] = static_route['description']
                if self.tag:
                    info_dict['tag'] = self.tag
                    exist_dict['tag'] = static_route['tag']
                if self.pref:
                    info_dict['preference'] = str(self.pref)
                    exist_dict['preference'] = static_route['preference']
                if self.nhp_interface:
                    if self.nhp_interface.lower() == 'invalid0':
                        info_dict['ifName'] = 'Invalid0'
                    else:
                        info_dict['ifName'] = 'Invalid0'
                    exist_dict['ifName'] = static_route['ifName']
                if self.next_hop:
                    info_dict['nexthop'] = self.next_hop
                    exist_dict['nexthop'] = static_route['nexthop']
                if self.bfd_session_name:
                    info_dict['bfdEnable'] = 'true'
                else:
                    info_dict['bfdEnable'] = 'false'
                exist_dict['bfdEnable'] = static_route['bfdEnable']
                if exist_dict != info_dict:
                    change = True
                else:
                    change = False
                change_list.append(change)
        if False in change_list:
            change = False
        else:
            change = True
        return change
    else:
        for static_route in self.static_routes_info['sroute']:
            if static_route['nexthop'] and self.next_hop:
                if static_route['prefix'].lower() == self.prefix.lower() and static_route['maskLength'] == self.mask and (static_route['nexthop'].lower() == self.next_hop.lower()) and (static_route['afType'] == version):
                    change = True
                    return change
            if static_route['ifName'] and self.nhp_interface:
                if static_route['prefix'].lower() == self.prefix.lower() and static_route['maskLength'] == self.mask and (static_route['ifName'].lower() == self.nhp_interface.lower()) and (static_route['afType'] == version):
                    change = True
                    return change
            else:
                continue
        change = False
    return change