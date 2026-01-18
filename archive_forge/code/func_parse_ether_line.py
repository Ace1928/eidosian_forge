from __future__ import (absolute_import, division, print_function)
import re
from ansible.module_utils.facts.network.base import NetworkCollector
from ansible.module_utils.facts.network.generic_bsd import GenericBsdIfconfigNetwork
def parse_ether_line(self, words, current_if, ips):
    macaddress = ''
    for octet in words[1].split(':'):
        octet = ('0' + octet)[-2:None]
        macaddress += octet + ':'
    current_if['macaddress'] = macaddress[0:-1]