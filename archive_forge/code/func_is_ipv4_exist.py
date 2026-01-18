from __future__ import (absolute_import, division, print_function)
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config, ce_argument_spec
def is_ipv4_exist(self, addr, maskstr, ipv4_type):
    """"Check IPv4 address exist"""
    addrs = self.intf_info['am4CfgAddr']
    if not addrs:
        return False
    for address in addrs:
        if address['ifIpAddr'] == addr:
            return address['subnetMask'] == maskstr and address['addrType'] == ipv4_type
    return False