from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.netconf import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.junipernetworks.junos.plugins.module_utils.network.junos.facts.facts import (
from ansible_collections.junipernetworks.junos.plugins.module_utils.network.junos.junos import (
def get_prefix_lists_facts(self, data=None):
    """Get the 'facts' (the current configuration)
        :rtype: A dictionary
        :returns: The current configuration as a dictionary
        """
    facts, _warnings = Facts(self._module).get_facts(self.gather_subset, self.gather_network_resources, data=data)
    prefix_lists_facts = facts['ansible_network_resources'].get('prefix_lists')
    if not prefix_lists_facts:
        return []
    return prefix_lists_facts