from __future__ import (absolute_import, division, print_function)
import re
from copy import deepcopy
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts import Facts
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible.module_utils.connection import ConnectionError
def get_vlans_common(self, vlans, cfg_vlans):
    """Returns the vlan ranges that are common in the want and have
        vlans lists
        """
    vlans = self.get_vlan_id_list(vlans)
    cfg_vlans = self.get_vlan_id_list(cfg_vlans)
    return self.get_vlan_range_list(list(set(vlans).intersection(set(cfg_vlans))))