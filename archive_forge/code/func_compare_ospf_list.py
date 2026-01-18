from __future__ import absolute_import, division, print_function
import re
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.arista.eos.plugins.module_utils.network.eos.facts.facts import Facts
def compare_ospf_list(self, w_list, h_list, l_key):
    return_list = []
    for w in w_list:
        present = False
        for h in h_list:
            diff = dict_diff(h, w)
            if not diff:
                present = True
                break
        if not present:
            return_list.append(w)
    return return_list