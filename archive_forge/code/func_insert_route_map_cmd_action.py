from __future__ import absolute_import, division, print_function
from copy import deepcopy
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts \
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils \
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
def insert_route_map_cmd_action(self, command, want):
    """Insert the "action" value into the specified "command" if it is not
        already present. This dictionary member will not be present in the
        command obtained from the "diff" utility if it is unchanged from its
        currently configured value because it is not a "difference" in the
        configuration requested by the playbook versus the current
        configuration. It is, however, needed in order to create the
        appropriate REST API for modifying other attributes in the route map."""
    conf_map_name = command.get('map_name', None)
    conf_seq_num = command.get('sequence_num', None)
    if not conf_map_name or not conf_seq_num:
        return
    conf_action = command.get('action', None)
    if conf_action:
        return
    matching_map_in_want = self.get_matching_map(conf_map_name, conf_seq_num, want)
    if matching_map_in_want:
        conf_action = matching_map_in_want.get('action')
        if conf_action is not None:
            command['action'] = conf_action