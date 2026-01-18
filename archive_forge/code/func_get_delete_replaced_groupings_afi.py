from __future__ import absolute_import, division, print_function
from copy import deepcopy
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts import Facts
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
def get_delete_replaced_groupings_afi(self, want_afi, have_afi):
    """creates and builds a list of requests to handle all parts that need to be deleted
        while handling the replaced state for an address family"""
    sent_commands = {}
    requests = []
    diff_requested = get_diff(have_afi, want_afi, self.test_keys)
    if diff_requested.get('vlans') and 'vlans' in want_afi:
        to_delete = {'afi': have_afi['afi'], 'vlans': diff_requested['vlans']}
        sent_commands['vlans'] = deepcopy(diff_requested['vlans'])
        requests.extend(self.get_delete_vlans_requests(to_delete))
    if diff_requested.get('trusted') and 'trusted' in want_afi:
        to_delete = {'afi': have_afi['afi'], 'trusted': diff_requested['trusted']}
        sent_commands['trusted'] = deepcopy(diff_requested['trusted'])
        requests.extend(self.get_delete_trusted_requests(to_delete))
    if diff_requested.get('source_bindings') and 'source_bindings' in want_afi:
        if want_afi['source_bindings'] == []:
            sent_commands['source_bindings'] = deepcopy(have_afi['source_bindings'])
            requests.extend(self.get_delete_specific_source_bindings_requests(have_afi))
        else:
            sent_commands['source_bindings'] = deepcopy(diff_requested['source_bindings'])
            for entry in diff_requested['source_bindings']:
                requests.extend(self.get_delete_individual_source_bindings_requests(have_afi, entry))
    return (sent_commands, requests)