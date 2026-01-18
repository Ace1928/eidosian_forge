from __future__ import (absolute_import, division, print_function)
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import string_types
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config
def operate_static_route_singlebfd(self, version, prefix, nhp_interface, next_hop, destvrf, state):
    """operate ipv4 static route singleBFD"""
    min_tx_interval = self.min_tx_interval
    min_rx_interval = self.min_rx_interval
    multiplier = self.detect_multiplier
    min_tx_interval_xml = '\n'
    min_rx_interval_xml = '\n'
    multiplier_xml = '\n'
    local_address_xml = '\n'
    if next_hop is None:
        next_hop = '0.0.0.0'
    if destvrf is None:
        dest_vpn_instance = '_public_'
    else:
        dest_vpn_instance = destvrf
    if nhp_interface is None:
        nhp_interface = 'Invalid0'
    if min_tx_interval is not None:
        min_tx_interval_xml = CE_NC_SET_IPV4_STATIC_ROUTE_BFDCOMMON_MINTX % min_tx_interval
    if min_rx_interval is not None:
        min_rx_interval_xml = CE_NC_SET_IPV4_STATIC_ROUTE_BFDCOMMON_MINRX % min_rx_interval
    if multiplier is not None:
        multiplier_xml = CE_NC_SET_IPV4_STATIC_ROUTE_BFDCOMMON_MUL % multiplier
    if prefix is not None:
        local_address_xml = CE_NC_SET_STATIC_ROUTE_SINGLEBFD_LOCALADRESS % prefix
    if state == 'present':
        configxmlstr = CE_NC_SET_STATIC_ROUTE_SINGLEBFD % (version, nhp_interface, dest_vpn_instance, next_hop, local_address_xml, min_tx_interval_xml, min_rx_interval_xml, multiplier_xml)
    else:
        configxmlstr = CE_NC_DELETE_STATIC_ROUTE_SINGLEBFD % (version, nhp_interface, dest_vpn_instance, next_hop)
    conf_str = build_config_xml(configxmlstr)
    recv_xml = set_nc_config(self.module, conf_str)
    self._checkresponse_(recv_xml, 'OPERATE_STATIC_ROUTE_singleBFD')