from __future__ import (absolute_import, division, print_function)
import re
from ansible.module_utils.facts.network.base import NetworkCollector
from ansible.module_utils.facts.network.generic_bsd import GenericBsdIfconfigNetwork
def parse_interface_line(self, words, current_if, interfaces):
    device = words[0][0:-1]
    if device not in interfaces:
        current_if = {'device': device, 'ipv4': [], 'ipv6': [], 'type': 'unknown'}
    else:
        current_if = interfaces[device]
    flags = self.get_options(words[1])
    v = 'ipv4'
    if 'IPv6' in flags:
        v = 'ipv6'
    if 'LOOPBACK' in flags:
        current_if['type'] = 'loopback'
    current_if[v].append({'flags': flags, 'mtu': words[3]})
    current_if['macaddress'] = 'unknown'
    return current_if