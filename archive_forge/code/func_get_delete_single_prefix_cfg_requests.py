from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts \
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils \
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
def get_delete_single_prefix_cfg_requests(self, command, have):
    """Create and return the appropriate set of REST API requests to delete
        the prefix set configuration specified by the current "command"."""
    requests = list()
    pfx_set_name = command.get('name', None)
    if not pfx_set_name:
        return requests
    cfg_prefix_set = self.prefix_set_in_config(pfx_set_name, have)
    if not cfg_prefix_set:
        return requests
    prefixes = command.get('prefixes', None)
    if not prefixes or prefixes == []:
        requests = self.get_delete_prefix_set_cfg(command)
    else:
        requests = self.get_delete_one_prefix_list_cfg(cfg_prefix_set, command)
    return requests