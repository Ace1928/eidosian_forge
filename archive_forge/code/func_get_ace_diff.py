from __future__ import absolute_import, division, print_function
import itertools
import re
import socket
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.arista.eos.plugins.module_utils.network.eos.facts.facts import Facts
def get_ace_diff(want_ace, have_ace):
    if not have_ace:
        return dict_diff({}, want_ace)
    for h_a in have_ace:
        d = dict_diff(want_ace, h_a)
        if not d:
            break
    return d