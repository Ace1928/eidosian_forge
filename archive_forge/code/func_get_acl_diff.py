from __future__ import absolute_import, division, print_function
import itertools
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.arista.eos.plugins.module_utils.network.eos.facts.facts import Facts
def get_acl_diff(self, w, h, intersection=False):
    diff_v4 = []
    diff_v6 = []
    w_acls_v4 = []
    w_acls_v6 = []
    h_acls_v4 = []
    h_acls_v6 = []
    for w_group in w['access_groups']:
        if w_group['afi'] == 'ipv4':
            w_acls_v4 = w_group['acls']
        if w_group['afi'] == 'ipv6':
            w_acls_v6 = w_group['acls']
    for h_group in h['access_groups']:
        if h_group['afi'] == 'ipv4':
            h_acls_v4 = h_group['acls']
        if h_group['afi'] == 'ipv6':
            h_acls_v6 = h_group['acls']
    for item in w_acls_v4:
        match = list(filter(lambda x: x['name'] == item['name'], h_acls_v4))
        if match:
            if item['direction'] == match[0]['direction']:
                if intersection:
                    diff_v4.append(item)
            elif not intersection:
                diff_v4.append(item)
        elif not intersection:
            diff_v4.append(item)
    for item in w_acls_v6:
        match = list(filter(lambda x: x['name'] == item['name'], h_acls_v6))
        if match:
            if item['direction'] == match[0]['direction']:
                if intersection:
                    diff_v6.append(item)
            elif not intersection:
                diff_v6.append(item)
        elif not intersection:
            diff_v6.append(item)
    return (diff_v4, diff_v6)