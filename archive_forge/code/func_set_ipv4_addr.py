from __future__ import (absolute_import, division, print_function)
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config, ce_argument_spec
def set_ipv4_addr(self, ifname, addr, mask, ipv4_type):
    """Set interface IPv4 address"""
    if not addr or not mask or (not type):
        return
    maskstr = self.convert_len_to_mask(mask)
    if self.state == 'present':
        if not self.is_ipv4_exist(addr, maskstr, ipv4_type):
            if ipv4_type == 'main':
                main_addr = self.get_ipv4_main_addr()
                if not main_addr:
                    xml_str = CE_NC_ADD_IPV4 % (ifname, addr, maskstr, ipv4_type)
                    self.netconf_set_config(xml_str, 'ADD_IPV4_ADDR')
                else:
                    xml_str = CE_NC_MERGE_IPV4 % (ifname, main_addr['ifIpAddr'], main_addr['subnetMask'], addr, maskstr)
                    self.netconf_set_config(xml_str, 'MERGE_IPV4_ADDR')
            else:
                xml_str = CE_NC_ADD_IPV4 % (ifname, addr, maskstr, ipv4_type)
                self.netconf_set_config(xml_str, 'ADD_IPV4_ADDR')
            self.updates_cmd.append('interface %s' % ifname)
            if ipv4_type == 'main':
                self.updates_cmd.append('ip address %s %s' % (addr, maskstr))
            else:
                self.updates_cmd.append('ip address %s %s sub' % (addr, maskstr))
            self.changed = True
    elif self.is_ipv4_exist(addr, maskstr, ipv4_type):
        xml_str = CE_NC_DEL_IPV4 % (ifname, addr, maskstr, ipv4_type)
        self.netconf_set_config(xml_str, 'DEL_IPV4_ADDR')
        self.updates_cmd.append('interface %s' % ifname)
        if ipv4_type == 'main':
            self.updates_cmd.append('undo ip address %s %s' % (addr, maskstr))
        else:
            self.updates_cmd.append('undo ip address %s %s sub' % (addr, maskstr))
        self.changed = True