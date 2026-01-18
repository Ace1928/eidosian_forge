from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts \
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils \
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
def get_delete_one_prefix_list_cfg(self, cfg_prefix_set, command):
    """Create the list of REST API prefix deletion requests needed for deletion
        of the the requested set of prefixes from the currently configured
        prefix set specified by "cfg_prefix_set"."""
    pfx_delete_cfg_list = list()
    prefixes = command.get('prefixes', None)
    for prefix in prefixes:
        pfx_delete_cfg = self.prefix_get_delete_single_prefix_cfg(prefix, cfg_prefix_set, command)
        if pfx_delete_cfg and len(pfx_delete_cfg) > 0:
            pfx_delete_cfg_list.append(pfx_delete_cfg)
    return pfx_delete_cfg_list