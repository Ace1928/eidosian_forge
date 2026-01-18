from __future__ import absolute_import, division, print_function
import traceback
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.formatted_diff_utils import (
from ansible.module_utils._text import to_native
from ansible.module_utils.connection import ConnectionError
def get_merge_commands_requests(self, want, have):
    """Returns the commands and requests necessary to merge the provided
        configurations into the current configuration
        """
    commands = []
    requests = []
    if not want:
        return (commands, requests)
    if have:
        diff = get_diff(want, have, TEST_KEYS)
    else:
        diff = want
    for cmd in diff:
        name = cmd['name']
        if name == 'eth0':
            continue
        if cmd.get('trunk') and cmd['trunk'].get('allowed_vlans'):
            match = next((cnf for cnf in have if cnf['name'] == name), None)
            if match:
                cmd['trunk']['allowed_vlans'] = self.get_trunk_allowed_vlans_diff(cmd, match)
                if not cmd['trunk']['allowed_vlans']:
                    cmd.pop('trunk')
        if cmd.get('access') or cmd.get('trunk'):
            commands.append(cmd)
    requests = self.get_create_l2_interface_requests(commands)
    return (commands, requests)