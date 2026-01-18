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
def get_delete_commands_requests_for_deleted(self, want, have):
    """Returns the commands and requests necessary to remove the current
        configuration of the provided objects when state is deleted
        """
    commands = []
    requests = []
    if not have:
        return (commands, requests)
    if not want:
        commands = [remove_empties(conf) for conf in have]
        requests = self.get_delete_all_switchport_requests(commands)
        return (commands, requests)
    for conf in want:
        name = conf['name']
        matched = next((cnf for cnf in have if cnf['name'] == name), None)
        if matched:
            if not conf.get('access') and (not conf.get('trunk')):
                command = {'name': name}
                if matched.get('access'):
                    command['access'] = matched['access']
                if matched.get('trunk'):
                    command['trunk'] = matched['trunk']
                commands.append(command)
                requests.extend(self.get_delete_all_switchport_requests([command]))
            else:
                command = {}
                if conf.get('access'):
                    access_match = matched.get('access')
                    if conf['access'].get('vlan'):
                        if access_match and access_match.get('vlan') == conf['access']['vlan']:
                            command['access'] = {'vlan': conf['access']['vlan']}
                            requests.append(self.get_access_delete_switchport_request(name))
                    elif access_match and access_match.get('vlan'):
                        command['access'] = {'vlan': access_match['vlan']}
                        requests.append(self.get_access_delete_switchport_request(name))
                if conf.get('trunk'):
                    if conf['trunk'].get('allowed_vlans'):
                        trunk_vlans_to_delete = self.get_trunk_allowed_vlans_common(conf, matched)
                        if trunk_vlans_to_delete:
                            command['trunk'] = {'allowed_vlans': trunk_vlans_to_delete}
                            requests.append(self.get_trunk_allowed_vlans_delete_switchport_request(name, command['trunk']['allowed_vlans']))
                    else:
                        trunk_match = matched.get('trunk')
                        if trunk_match and trunk_match.get('allowed_vlans'):
                            command['trunk'] = {'allowed_vlans': trunk_match['allowed_vlans'].copy()}
                            requests.append(self.get_trunk_allowed_vlans_delete_switchport_request(name, command['trunk']['allowed_vlans']))
                if command:
                    command['name'] = name
                    commands.append(command)
    return (commands, requests)