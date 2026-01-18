from __future__ import absolute_import, division, print_function
import platform
import re
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.vyos import (
def parse_neighbors(self, data):
    facts = dict()
    for item in data:
        interface = self.parse_interface(item)
        host = self.parse_host(item)
        port = self.parse_port(item)
        if interface not in facts:
            facts[interface] = list()
        facts[interface].append(dict(host=host, port=port))
    return facts