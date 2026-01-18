from __future__ import (absolute_import, division, print_function)
import re
from copy import deepcopy
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts import Facts
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible.module_utils.connection import ConnectionError
def get_modify_stp_rapid_pvst_request(self, commands):
    request = None
    rapid_pvst = commands.get('rapid_pvst', None)
    if rapid_pvst:
        vlans_list = self.get_vlans_list(rapid_pvst)
        if vlans_list:
            url = '%s/rapid-pvst' % STP_PATH
            payload = {'openconfig-spanning-tree:rapid-pvst': {'vlan': vlans_list}}
            request = {'path': url, 'method': PATCH, 'data': payload}
    return request