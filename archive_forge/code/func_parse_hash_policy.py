from __future__ import absolute_import, division, print_function
from copy import deepcopy
from re import M, findall, search
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import utils
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.argspec.lag_interfaces.lag_interfaces import (
def parse_hash_policy(self, conf):
    hash_policy = None
    if conf:
        hash_policy = search('^.*hash-policy (.+)', conf, M)
        hash_policy = hash_policy.group(1).strip("'")
    return hash_policy