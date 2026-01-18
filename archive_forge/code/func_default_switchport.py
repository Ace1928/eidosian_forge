from __future__ import (absolute_import, division, print_function)
import re
from xml.etree import ElementTree as ET
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config, ce_argument_spec
def default_switchport(self, ifname):
    """Set interface default or unconfigured"""
    change = False
    if self.intf_info['linkType'] != 'access':
        self.updates_cmd.append('interface %s' % ifname)
        self.updates_cmd.append('port link-type access')
        self.updates_cmd.append('port default vlan 1')
        change = True
    elif self.intf_info['pvid'] != '1':
        self.updates_cmd.append('interface %s' % ifname)
        self.updates_cmd.append('port default vlan 1')
        change = True
    if not change:
        return
    conf_str = CE_NC_SET_DEFAULT_PORT % ifname
    rcv_xml = set_nc_config(self.module, conf_str)
    self.check_response(rcv_xml, 'DEFAULT_INTF_VLAN')
    self.changed = True