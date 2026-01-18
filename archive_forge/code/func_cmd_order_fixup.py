from __future__ import absolute_import, division, print_function
import re
from copy import deepcopy
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.facts.facts import Facts
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.utils.utils import (
def cmd_order_fixup(self, cmds, name):
    """Inserts 'interface <name>' config at the beginning of populated command list; reorders dependent commands that must process after others."""
    if cmds:
        if name and (not [item for item in cmds if item.startswith('interface')]):
            cmds.insert(0, 'interface ' + name)
        redirects = [item for item in cmds if re.match('(no )*ip(v6)* redirects', item)]
        if redirects:
            redirects = redirects.pop()
            cmds.remove(redirects)
            cmds.append(redirects)