from __future__ import absolute_import, division, print_function
import re
from copy import deepcopy
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts import Facts
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible.module_utils.connection import ConnectionError
def get_vlan_range_diff(self, config_vlans, match_vlans):
    """Returns the vlan ranges present only in 'config_vlans'
        and not in 'match_vlans' in vlans spec format
        """
    if not config_vlans:
        return []
    if not match_vlans:
        return config_vlans
    config_vlans = self.get_vlan_id_list(config_vlans)
    match_vlans = self.get_vlan_id_list(match_vlans)
    return self.get_vlan_range_list(list(set(config_vlans) - set(match_vlans)))