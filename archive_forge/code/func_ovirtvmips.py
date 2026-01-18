from __future__ import (absolute_import, division, print_function)
from xml.etree import ElementTree
def ovirtvmips(self, ovirt_vms, attr=None, network_ip=None):
    """Return list of IPs"""
    return self._parse_ips(ovirt_vms, attr=attr)