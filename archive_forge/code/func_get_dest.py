from __future__ import absolute_import, division, print_function
import re
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.arista.eos.plugins.module_utils.network.eos.facts.facts import Facts
def get_dest(config):
    dest = []
    for c in config:
        for address_family in c['address_families']:
            for route in address_family['routes']:
                dest.append(route['dest'])
    return dest