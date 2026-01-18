from __future__ import (absolute_import, division, print_function)
from xml.etree import ElementTree
def ovirtvmipsv4(self, ovirt_vms, attr=None, network_ip=None):
    """Return list of IPv4 IPs"""
    ips = self._parse_ips(ovirt_vms, lambda version: version == 'v4', attr)
    if attr:
        return dict(((k, list(filter(lambda x: self.__address_in_network(x, network_ip), v))) for k, v in ips.items()))
    return list(filter(lambda x: self.__address_in_network(x, network_ip), ips))