from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts \
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils \
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
def prefix_in_prefix_list_cfg(self, prefix, cfg_prefix_set):
    """Determine, based on the keys, if the "target" prefix specified by the "prefix"
        input parameter is present in the currently configured prefix set specified
        ty the "cfg_prefix_set" input parameter. Return "True" if the prifix is found,
        or "False" if it isn't."""
    req_pfx = prefix.get('prefix', None)
    req_seq = prefix.get('sequence', None)
    req_ge = prefix.get('ge', None)
    req_le = prefix.get('le', None)
    cfg_prefix_list = cfg_prefix_set.get('prefixes', None)
    if not cfg_prefix_list:
        return False
    for cfg_prefix in cfg_prefix_list:
        cfg_pfx = cfg_prefix.get('prefix', None)
        cfg_seq = cfg_prefix.get('sequence', None)
        cfg_ge = cfg_prefix.get('ge', None)
        cfg_le = cfg_prefix.get('le', None)
        if not (req_pfx and cfg_pfx and (req_pfx == cfg_pfx)):
            continue
        if not (req_seq and cfg_seq and (req_seq == cfg_seq)):
            continue
        if not req_ge:
            if cfg_ge:
                continue
        elif not cfg_ge or req_ge != cfg_ge:
            continue
        if not req_le:
            if cfg_le:
                continue
        elif not cfg_le or req_le != cfg_le:
            continue
        return True
    return False