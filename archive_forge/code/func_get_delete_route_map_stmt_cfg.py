from __future__ import absolute_import, division, print_function
from copy import deepcopy
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts \
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils \
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
def get_delete_route_map_stmt_cfg(self, command, requests):
    """Append to the input list of REST API requests the REST API to
        delete all configuration for the route map "statement" (route
        map sub-section) specified by the combination of the route
        map name and "statement" sequence number in the input
        "command" dict."""
    conf_map_name = command.get('map_name')
    conf_seq_num = command.get('sequence_num')
    req_seq_num = str(conf_seq_num)
    delete_rmap_stmt_path = self.route_map_stmt_uri.format(conf_map_name, req_seq_num)
    request = {'path': delete_rmap_stmt_path, 'method': DELETE}
    requests.append(request)