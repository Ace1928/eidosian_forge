from __future__ import absolute_import, division, print_function
from copy import deepcopy
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts \
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils \
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
def get_delete_one_route_map_cfg(self, conf_map_name, requests):
    """Append to the input list of REST API requests the REST API to
        delete all configuration for the specified route map."""
    delete_rmap_path = self.route_map_uri.format(conf_map_name)
    request = {'path': delete_rmap_path, 'method': DELETE}
    requests.append(request)