from __future__ import absolute_import, division, print_function
import platform
import re
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import iteritems
from ansible_collections.frr.frr.plugins.module_utils.network.frr.frr import (
def add_ip_address(self, address, family):
    if family == 'ipv4':
        self.facts['all_ipv4_addresses'].append(address)
    else:
        self.facts['all_ipv6_addresses'].append(address)