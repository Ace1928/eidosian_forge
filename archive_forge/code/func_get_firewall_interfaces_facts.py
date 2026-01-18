from __future__ import absolute_import, division, print_function
from copy import deepcopy
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.facts.facts import Facts
def get_firewall_interfaces_facts(self, data=None):
    """Get the 'facts' (the current configuration)

        :rtype: A dictionary
        :returns: The current configuration as a dictionary
        """
    facts, _warnings = Facts(self._module).get_facts(self.gather_subset, self.gather_network_resources, data=data)
    firewall_interfaces_facts = facts['ansible_network_resources'].get('firewall_interfaces')
    if not firewall_interfaces_facts:
        return []
    return firewall_interfaces_facts