from __future__ import absolute_import, division, print_function
import itertools
import re
import socket
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.arista.eos.plugins.module_utils.network.eos.facts.facts import Facts
def get_updated_ace(w, h):
    if not dict_diff(w, h):
        return
    w_updated = w.copy()
    for hkey in h.keys():
        if hkey not in w.keys():
            w_updated.update({hkey: h[hkey]})
        else:
            w_updated.update({hkey: w[hkey]})
    return w_updated