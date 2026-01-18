from __future__ import absolute_import, division, print_function
from copy import deepcopy
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts import Facts
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.formatted_diff_utils import (
from ansible.module_utils.connection import ConnectionError
def get_create_requests(self, configs, have):
    requests = []
    if not configs:
        return requests
    requests_vrf = self.get_create_vrf_requests(configs, have)
    if requests_vrf:
        requests.extend(requests_vrf)
    requests_vrf_intf = self.get_create_vrf_interface_requests(configs, have)
    if requests_vrf_intf:
        requests.extend(requests_vrf_intf)
    return requests