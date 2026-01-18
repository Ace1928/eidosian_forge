from __future__ import absolute_import, division, print_function
from copy import deepcopy
from re import M, findall
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import utils
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.argspec.lldp_global.lldp_global import (
def parse_protocols(self, conf):
    protocol_support = None
    if conf:
        protocols = findall('^.*legacy-protocols (.+)', conf, M)
        if protocols:
            protocol_support = []
            for protocol in protocols:
                protocol_support.append(protocol.strip("'"))
    return protocol_support