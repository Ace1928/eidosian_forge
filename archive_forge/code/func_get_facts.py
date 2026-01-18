from __future__ import (absolute_import, division, print_function)
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.argspec.system.system import SystemArgs
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.facts.facts import FactsBase
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.facts.system.system import SystemFacts
def get_facts(self, facts_type=None, data=None):
    """ Collect the facts for fortios
        :param facts_type: List of facts types
        :param data: previously collected conf
        :rtype: dict
        :return: the facts gathered
        """
    self.get_network_legacy_facts(self.FACT_SUBSETS, facts_type)
    return (self.ansible_facts, self._warnings)