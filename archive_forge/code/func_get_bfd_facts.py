from __future__ import (absolute_import, division, print_function)
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts import Facts
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from copy import deepcopy
def get_bfd_facts(self):
    """ Get the 'facts' (the current configuration)

        :rtype: A dictionary
        :returns: The current configuration as a dictionary
        """
    facts, _warnings = Facts(self._module).get_facts(self.gather_subset, self.gather_network_resources)
    bfd_facts = facts['ansible_network_resources'].get('bfd')
    if not bfd_facts:
        return {}
    return bfd_facts