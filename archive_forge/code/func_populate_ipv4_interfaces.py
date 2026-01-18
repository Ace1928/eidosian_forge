from __future__ import absolute_import, division, print_function
import platform
import re
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import iteritems
from ansible_collections.frr.frr.plugins.module_utils.network.frr.frr import (
def populate_ipv4_interfaces(self, data):
    for key, value in data.items():
        self.facts['interfaces'][key]['ipv4'] = list()
        primary_address = addresses = []
        primary_address = re.findall('inet (\\S+) broadcast (?:\\S+)(?:\\s{2,})', value, re.M)
        addresses = re.findall('inet (\\S+) broadcast (?:\\S+)(?:\\s+)secondary', value, re.M)
        if len(primary_address) == 0:
            continue
        addresses.append(primary_address[0])
        for address in addresses:
            addr, subnet = address.split('/')
            ipv4 = dict(address=addr.strip(), subnet=subnet.strip())
            self.add_ip_address(addr.strip(), 'ipv4')
            self.facts['interfaces'][key]['ipv4'].append(ipv4)