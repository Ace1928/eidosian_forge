from __future__ import absolute_import, division, print_function
import re
from copy import deepcopy
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.facts.facts import Facts
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.utils.utils import (
def del_all_attribs(self, obj):
    commands = []
    if not obj or len(obj.keys()) == 1:
        return commands
    commands = self.generate_delete_commands(obj)
    self.cmd_order_fixup(commands, obj['name'])
    return commands