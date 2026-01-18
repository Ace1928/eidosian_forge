from __future__ import (absolute_import, division, print_function)
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import string_types
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config
def operate_static_route_globalbfd(self):
    """set globalbfd update command"""
    min_tx_interval = self.min_tx_interval
    min_rx_interval = self.min_rx_interval
    multiplier = self.detect_multiplier
    min_tx_interval_xml = '\n'
    min_rx_interval_xml = '\n'
    multiplier_xml = '\n'
    if self.state == 'present':
        if min_tx_interval is not None:
            min_tx_interval_xml = CE_NC_SET_IPV4_STATIC_ROUTE_BFDCOMMON_MINTX % min_tx_interval
        if min_rx_interval is not None:
            min_rx_interval_xml = CE_NC_SET_IPV4_STATIC_ROUTE_BFDCOMMON_MINRX % min_rx_interval
        if multiplier is not None:
            multiplier_xml = CE_NC_SET_IPV4_STATIC_ROUTE_BFDCOMMON_MUL % multiplier
        configxmlstr = CE_NC_SET_IPV4_STATIC_ROUTE_GLOBALBFD % (min_tx_interval_xml, min_rx_interval_xml, multiplier_xml)
        conf_str = build_config_xml(configxmlstr)
        recv_xml = set_nc_config(self.module, conf_str)
        self._checkresponse_(recv_xml, 'OPERATE_STATIC_ROUTE_globalBFD')
    if self.state == 'absent' and self.commands:
        min_tx_interval_xml = CE_NC_SET_IPV4_STATIC_ROUTE_BFDCOMMON_MINTX % 1000
        min_rx_interval_xml = CE_NC_SET_IPV4_STATIC_ROUTE_BFDCOMMON_MINRX % 1000
        multiplier_xml = CE_NC_SET_IPV4_STATIC_ROUTE_BFDCOMMON_MUL % 3
        configxmlstr = CE_NC_SET_IPV4_STATIC_ROUTE_GLOBALBFD % (min_tx_interval_xml, min_rx_interval_xml, multiplier_xml)
        conf_str = build_config_xml(configxmlstr)
        recv_xml = set_nc_config(self.module, conf_str)
        self._checkresponse_(recv_xml, 'OPERATE_STATIC_ROUTE_globalBFD')