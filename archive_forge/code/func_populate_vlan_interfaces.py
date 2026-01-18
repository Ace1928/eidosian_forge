from __future__ import absolute_import, division, print_function
import re
import json
from ansible_collections.community.network.plugins.module_utils.network.exos.exos import run_commands
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import iteritems
def populate_vlan_interfaces(self, data, sysmac):
    for elem in data:
        if 'vlanProc' in elem:
            key = elem['vlanProc']['name1']
            if key not in self.facts['interfaces']:
                intf = dict()
                intf['type'] = 'VLAN'
                intf['macaddress'] = sysmac
                self.facts['interfaces'][key] = intf
            if elem['vlanProc']['ipAddress'] != '0.0.0.0':
                self.facts['interfaces'][key]['ipv4'] = list()
                addr = elem['vlanProc']['ipAddress']
                subnet = elem['vlanProc']['maskForDisplay']
                ipv4 = dict(address=addr, subnet=subnet)
                self.add_ip_address(addr, 'ipv4')
                self.facts['interfaces'][key]['ipv4'].append(ipv4)
        if 'rtifIpv6Address' in elem:
            key = elem['rtifIpv6Address']['rtifName']
            if key not in self.facts['interfaces']:
                intf = dict()
                intf['type'] = 'VLAN'
                intf['macaddress'] = sysmac
                self.facts['interfaces'][key] = intf
            self.facts['interfaces'][key]['ipv6'] = list()
            addr, subnet = elem['rtifIpv6Address']['ipv6_address_mask'].split('/')
            ipv6 = dict(address=addr, subnet=subnet)
            self.add_ip_address(addr, 'ipv6')
            self.facts['interfaces'][key]['ipv6'].append(ipv6)