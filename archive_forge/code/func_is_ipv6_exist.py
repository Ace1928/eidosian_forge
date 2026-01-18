from __future__ import (absolute_import, division, print_function)
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config, ce_argument_spec
def is_ipv6_exist(self, addr, masklen):
    """Check IPv6 address exist"""
    addrs = self.intf_info['am6CfgAddr']
    if not addrs:
        return False
    for address in addrs:
        if address['ifIp6Addr'] == addr.upper():
            if address['addrPrefixLen'] == masklen and address['addrType6'] == 'global':
                return True
            else:
                self.module.fail_json(msg='Error: Input IPv6 address or mask is invalid.')
    return False