from __future__ import absolute_import, division, print_function
from copy import deepcopy
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts import Facts
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
def prep_replaced_to_merge(self, diff, afis):
    """preps results from a get diff for use in merging. needed for source bindings to have all data needed. get diff only returns the fields that
        are different in each source binding when all data for it is needed instead. Fills in each source binding in diff with what is found for it in afis"""
    if not diff or not diff.get('afis'):
        return {}
    for diff_afi in diff['afis']:
        if 'source_bindings' in diff_afi:
            for binding in diff_afi['source_bindings']:
                binding.update(self.match_binding(binding['mac_addr'], afis['want_' + diff_afi['afi']]['source_bindings']))