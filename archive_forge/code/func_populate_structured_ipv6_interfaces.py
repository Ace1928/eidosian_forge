from __future__ import absolute_import, division, print_function
import platform
import re
from ansible.module_utils.six import iteritems
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.nxos import (
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.utils.utils import (
def populate_structured_ipv6_interfaces(self, data):
    try:
        data = data['TABLE_intf']
        if data:
            if isinstance(data, dict):
                data = [data]
            for item in data:
                name = item['ROW_intf']['intf-name']
                intf = self.facts['interfaces'][name]
                intf['ipv6'] = self.transform_dict(item, self.INTERFACE_IPV6_MAP)
                try:
                    addr = item['ROW_intf']['addr']
                except KeyError:
                    addr = item['ROW_intf']['TABLE_addr']['ROW_addr']['addr']
                self.facts['all_ipv6_addresses'].append(addr)
        else:
            return ''
    except TypeError:
        return ''