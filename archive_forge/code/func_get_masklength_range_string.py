from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts \
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils \
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
def get_masklength_range_string(self, pfx_ge, pfx_le, prefix_net):
    """Determine the "masklength range" string required for the openconfig
        REST API to configure the affected prefix."""
    if not pfx_ge and (not pfx_le):
        masklength_range_string = 'exact'
    elif pfx_ge and (not pfx_le):
        masklength_range_string = str(pfx_ge) + '..' + str(prefix_net['max_prefixlen'])
    elif not pfx_ge and pfx_le:
        masklength_range_string = str(prefix_net['prefixlen']) + '..' + str(pfx_le)
    else:
        masklength_range_string = str(pfx_ge) + '..' + str(pfx_le)
    return masklength_range_string