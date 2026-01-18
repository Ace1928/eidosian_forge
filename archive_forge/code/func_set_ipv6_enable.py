from __future__ import (absolute_import, division, print_function)
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config, ce_argument_spec
def set_ipv6_enable(self, ifname):
    """Set interface IPv6 enable"""
    if self.state == 'present':
        if self.intf_info['enableFlag'] == 'false':
            xml_str = CE_NC_MERGE_IPV6_ENABLE % (ifname, 'true')
            self.netconf_set_config(xml_str, 'SET_IPV6_ENABLE')
            self.updates_cmd.append('interface %s' % ifname)
            self.updates_cmd.append('ipv6 enable')
            self.changed = True
    elif self.intf_info['enableFlag'] == 'true':
        xml_str = CE_NC_MERGE_IPV6_ENABLE % (ifname, 'false')
        self.netconf_set_config(xml_str, 'SET_IPV6_DISABLE')
        self.updates_cmd.append('interface %s' % ifname)
        self.updates_cmd.append('undo ipv6 enable')
        self.changed = True