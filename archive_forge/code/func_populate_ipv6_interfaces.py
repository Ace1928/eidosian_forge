from __future__ import absolute_import, division, print_function
import platform
import re
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import iteritems
from ansible_collections.frr.frr.plugins.module_utils.network.frr.frr import (
def populate_ipv6_interfaces(self, data):
    for key, value in data.items():
        self.facts['interfaces'][key]['ipv6'] = list()
        addresses = re.findall('inet6 (\\S+)', value, re.M)
        for address in addresses:
            addr, subnet = address.split('/')
            ipv6 = dict(address=addr.strip(), subnet=subnet.strip())
            self.add_ip_address(addr.strip(), 'ipv6')
            self.facts['interfaces'][key]['ipv6'].append(ipv6)