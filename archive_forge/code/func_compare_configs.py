from __future__ import absolute_import, division, print_function
import itertools
import re
import socket
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.arista.eos.plugins.module_utils.network.eos.facts.facts import Facts
def compare_configs(self, have, want):
    commands = []
    want = list(itertools.chain(*want))
    have = list(itertools.chain(*have))
    h_index = 0
    config = list(want)
    for w in want:
        access_list = re.findall('(ip.*) access-list (.*)', w)
        if access_list:
            if w in have:
                h_index = have.index(w)
        else:
            for num, h in enumerate(have, start=h_index + 1):
                if 'access-list' not in h:
                    seq_num = re.search('(\\d+) (.*)', w)
                    if seq_num:
                        have_seq_num = re.search('(\\d+) (.*)', h)
                        if seq_num.group(1) == have_seq_num.group(1) and have_seq_num.group(2) != seq_num.group(2):
                            negate_cmd = 'no ' + seq_num.group(1)
                            config.insert(config.index(w), negate_cmd)
                    if w in h:
                        config.pop(config.index(w))
                        break
    for c in config:
        access_list = re.findall('(ip.*) access-list (.*)', c)
        if access_list:
            acl_index = config.index(c)
        else:
            if config[acl_index] not in commands:
                commands.append(config[acl_index])
            commands.append(c)
    return commands