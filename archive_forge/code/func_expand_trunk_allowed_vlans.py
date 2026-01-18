from __future__ import absolute_import, division, print_function
import re
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.facts.facts import Facts
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.utils.utils import (
def expand_trunk_allowed_vlans(self, d):
    if not d:
        return None
    if 'trunk' in d and d['trunk']:
        if 'allowed_vlans' in d['trunk']:
            allowed_vlans = vlan_range_to_list(d['trunk']['allowed_vlans'])
            vlans_list = [str(l) for l in sorted(allowed_vlans)]
            d['trunk']['allowed_vlans'] = ','.join(vlans_list)