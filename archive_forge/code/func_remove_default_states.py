from __future__ import absolute_import, division, print_function
import re
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.facts.facts import Facts
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.utils.utils import (
def remove_default_states(self, obj):
    """Removes non-empty but default states from the obj."""
    default_states = {'enabled': True, 'state': 'active', 'mode': 'ce'}
    for k in default_states.keys():
        if obj.get(k) == default_states[k]:
            obj.pop(k, None)
    return obj