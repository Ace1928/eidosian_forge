from __future__ import absolute_import, division, print_function
import re
from copy import deepcopy
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.facts.facts import Facts
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.utils.utils import (
def generate_afi_commands(self, diff):
    cmds = []
    for i in diff:
        cmd = 'ipv6 address ' if re.search('::', i['address']) else 'ip address '
        cmd += i['address']
        if i.get('secondary'):
            cmd += ' secondary'
        if i.get('tag'):
            cmd += ' tag ' + str(i['tag'])
        cmds.append(cmd)
    return cmds