from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.facts.facts import Facts
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.utils.utils import (
def process_access_group(self, item, deleted=False):
    commands = []
    for ag in item['access_groups']:
        ip = 'ipv6'
        if ag['afi'] == 'ipv4':
            ip = 'ip'
        if ag.get('acls'):
            commands.extend(self.process_acl(ag['acls'], ip, deleted))
    return commands