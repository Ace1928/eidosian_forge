from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts import Facts
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible.module_utils.connection import ConnectionError
def get_delete_lldp_global_completely_requests(self, have):
    """Get requests to delete all existing LLDP global
        configurations in the chassis
        """
    default_config_dict = {'enable': True, 'tlv_select': {'management_address': True, 'system_capabilities': True}}
    requests = []
    if default_config_dict != have:
        return [{'path': self.lldp_global_path, 'method': DELETE}]
    return requests