from __future__ import (absolute_import, division, print_function)
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import string_types
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config
def get_single_bfd(self, state):
    """get ipv4 sigle bfd"""
    self.static_routes_info['sroute_single_bfd'] = list()
    if self.aftype == 'v4':
        version = 'ipv4unicast'
    else:
        version = 'ipv6unicast'
    if state == 'absent':
        getbfdxmlstr = CE_NC_GET_STATIC_ROUTE_BFD_ABSENT % (version, self.nhp_interface, self.destvrf, self.next_hop)
    else:
        getbfdxmlstr = CE_NC_GET_STATIC_ROUTE_BFD % (version, self.nhp_interface, self.destvrf, self.next_hop)
    xml_bfd_str = get_nc_config(self.module, getbfdxmlstr)
    if 'data/' in xml_bfd_str:
        return
    xml_bfd_str = xml_bfd_str.replace('\r', '').replace('\n', '').replace('xmlns="urn:ietf:params:xml:ns:netconf:base:1.0"', '').replace('xmlns="http://www.huawei.com/netconf/vrp"', '')
    root = ElementTree.fromstring(xml_bfd_str)
    static_routes_bfd = root.findall('staticrt/staticrtbase/srBfdParas/srBfdPara')
    if static_routes_bfd:
        for static_route in static_routes_bfd:
            static_info = dict()
            for static_ele in static_route:
                if static_ele.tag in ['afType', 'destVrfName', 'nexthop', 'ifName']:
                    static_info[static_ele.tag] = static_ele.text
                if static_ele.tag == 'localAddress':
                    if static_ele.text is not None:
                        static_info['localAddress'] = static_ele.text
                    else:
                        static_info['localAddress'] = 'None'
                if static_ele.tag == 'minTxInterval':
                    if static_ele.text is not None:
                        static_info['minTxInterval'] = static_ele.text
                if static_ele.tag == 'minRxInterval':
                    if static_ele.text is not None:
                        static_info['minRxInterval'] = static_ele.text
                if static_ele.tag == 'multiplier':
                    if static_ele.text is not None:
                        static_info['multiplier'] = static_ele.text
            self.static_routes_info['sroute_single_bfd'].append(static_info)