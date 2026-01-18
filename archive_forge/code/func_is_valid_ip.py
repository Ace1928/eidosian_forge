from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ipaddress import ip_interface, ip_network
def is_valid_ip(addr, type='all'):
    if type in ['all', 'ipv4']:
        if validate_ip_address(addr):
            return True
    if type in ['all', 'ipv6']:
        if validate_ip_v6_address(addr):
            return True
    return False