from __future__ import absolute_import, division, print_function
from copy import deepcopy
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts \
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils \
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
def get_delete_single_route_map_requests(self, have, command, requests):
    """Create and return the appropriate set of route map REST APIs
        to delete the eligible requestd attributes from the  route map
        configuration specified by the current "command"."""
    if not command:
        return
    conf_map_name = command.get('map_name', None)
    if not conf_map_name:
        command = {}
        return
    conf_seq_num = command.get('sequence_num', None)
    if not conf_seq_num:
        if self.any_rmap_inst_in_have(conf_map_name, have):
            self.get_delete_one_route_map_cfg(conf_map_name, requests)
        return
    cmd_rmap_have = self.get_matching_map(conf_map_name, conf_seq_num, have)
    if not cmd_rmap_have:
        command = {}
        return
    cmd_match_top = command.get('match')
    if cmd_match_top:
        cmd_match_top = command['match']
    cmd_set_top = command.get('set')
    if cmd_set_top:
        cmd_set_top = command['set']
    if not cmd_match_top and (not cmd_set_top):
        self.get_delete_route_map_stmt_cfg(command, requests)
        return
    conf_action = command.get('action', None)
    if not conf_action:
        self._module.fail_json(msg="\nThe 'action' attribute is required, but is absentfor route map {0} sequence number {1}\n".format(conf_map_name, conf_seq_num))
    if conf_action not in ('permit', 'deny'):
        self._module.fail_json(msg="\nInvalid 'action' attribute value {0} forroute map {1} sequence number {2}\n".format(conf_action, conf_map_name, conf_seq_num))
        command = {}
        return
    if cmd_match_top:
        self.get_route_map_delete_match_attr(command, cmd_rmap_have, requests)
    if cmd_set_top:
        self.get_route_map_delete_set_attr(command, cmd_rmap_have, requests)
    if command:
        self.get_route_map_delete_call_attr(command, cmd_rmap_have, requests)
    return