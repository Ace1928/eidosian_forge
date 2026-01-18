from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.arista.eos.plugins.module_utils.network.eos.facts.facts import Facts
def state_replaced(want, have):
    """The command generator when state is replaced

    :rtype: A list
    :returns: the commands necessary to migrate the current configuration
              to the desired configuration
    """
    commands = set()
    commands.update(state_merged(want, have))
    commands.update(state_deleted(want, have))
    return list(commands)