from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import to_list
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.cmdref.telemetry.telemetry import (
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.facts.facts import Facts
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.nxos import NxosCmdRef
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.utils.telemetry.telemetry import (
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.utils.utils import (
def get_telemetry_facts(self, data=None):
    """Get the 'facts' (the current configuration)
        :rtype: A dictionary
        :returns: The current configuration as a dictionary
        """
    facts, _warnings = Facts(self._module).get_facts(self.gather_subset, self.gather_network_resources, data=data)
    telemetry_facts = facts['ansible_network_resources'].get('telemetry')
    if not telemetry_facts:
        return {}
    return telemetry_facts