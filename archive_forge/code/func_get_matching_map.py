from __future__ import absolute_import, division, print_function
from copy import deepcopy
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts \
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils \
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
@staticmethod
def get_matching_map(conf_map_name, conf_seq_num, input_list):
    """In the input list of command or configuration dicts, find the route map
        configuration "statement" (if it exists) for the specified map name
        and sequence number."""
    for cfg_route_map in input_list:
        if cfg_route_map.get('map_name') and cfg_route_map.get('sequence_num'):
            if cfg_route_map['map_name'] == conf_map_name and cfg_route_map.get('sequence_num') == conf_seq_num:
                return cfg_route_map
    return {}