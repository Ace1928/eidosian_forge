from __future__ import (absolute_import, division, print_function)
import re
from xml.etree import ElementTree as ET
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config, ce_argument_spec
def merge_dot1qtunnel_vlan(self, ifname, default_vlan):
    """Merge dot1qtunnel"""
    change = False
    conf_str = ''
    self.updates_cmd.append('interface %s' % ifname)
    if self.state == 'present':
        if self.intf_info['linkType'] == 'dot1qtunnel':
            if default_vlan and self.intf_info['pvid'] != default_vlan:
                self.updates_cmd.append('port default vlan %s' % default_vlan)
                conf_str = CE_NC_SET_PORT % (ifname, 'dot1qtunnel', default_vlan, '', '')
                change = True
        else:
            self.updates_cmd.append('port link-type dot1qtunnel')
            if default_vlan:
                self.updates_cmd.append('port default vlan %s' % default_vlan)
                conf_str = CE_NC_SET_PORT % (ifname, 'dot1qtunnel', default_vlan, '', '')
            else:
                conf_str = CE_NC_SET_PORT % (ifname, 'dot1qtunnel', '1', '', '')
            change = True
    elif self.state == 'absent':
        if self.intf_info['linkType'] == 'dot1qtunnel':
            if default_vlan and self.intf_info['pvid'] == default_vlan and (default_vlan != '1'):
                self.updates_cmd.append('undo port default vlan %s' % default_vlan)
                conf_str = CE_NC_SET_PORT % (ifname, 'dot1qtunnel', '1', '', '')
                change = True
    if not change:
        self.updates_cmd.pop()
        return
    conf_str = '<config>' + conf_str + '</config>'
    rcv_xml = set_nc_config(self.module, conf_str)
    self.check_response(rcv_xml, 'MERGE_DOT1QTUNNEL_PORT')
    self.changed = True