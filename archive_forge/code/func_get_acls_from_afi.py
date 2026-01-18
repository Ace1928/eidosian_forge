from __future__ import absolute_import, division, print_function
import itertools
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.arista.eos.plugins.module_utils.network.eos.facts.facts import Facts
def get_acls_from_afi(self, interface, afi, have):
    config = []
    for h in have:
        if h['name'] == interface:
            if 'access_groups' not in h.keys() or not h['access_groups']:
                continue
            if h['access_groups']:
                for h_grp in h['access_groups']:
                    if h_grp['afi'] == afi:
                        config = h_grp['acls']
    return config