from __future__ import absolute_import, division, print_function
import platform
import re
from ansible.module_utils.six import iteritems
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.nxos import (
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.utils.utils import (
def populate_structured_interfaces(self, data):
    interfaces = dict()
    data = data['TABLE_interface']['ROW_interface']
    if isinstance(data, dict):
        data = [data]
    for item in data:
        name = item['interface']
        intf = dict()
        if any((key.startswith('svi_') for key in item)):
            intf.update(self.transform_dict(item, self.INTERFACE_SVI_MAP))
        else:
            intf.update(self.transform_dict(item, self.INTERFACE_MAP))
        if 'eth_ip_addr' in item:
            intf['ipv4'] = self.transform_dict(item, self.INTERFACE_IPV4_MAP)
            self.facts['all_ipv4_addresses'].append(item['eth_ip_addr'])
        if 'svi_ip_addr' in item:
            intf['ipv4'] = self.transform_dict(item, self.INTERFACE_SVI_IPV4_MAP)
            self.facts['all_ipv4_addresses'].append(item['svi_ip_addr'])
        interfaces[name] = intf
    return interfaces