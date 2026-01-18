from __future__ import (absolute_import, division, print_function)
import re
from copy import deepcopy
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts import Facts
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible.module_utils.connection import ConnectionError
def get_modify_stp_requests(self, commands, have):
    requests = []
    if not commands:
        return requests
    global_request = self.get_modify_stp_global_request(commands, have)
    interfaces_request = self.get_modify_stp_interfaces_request(commands)
    mstp_requests = self.get_modify_stp_mstp_request(commands, have)
    pvst_request = self.get_modify_stp_pvst_request(commands)
    rapid_pvst_request = self.get_modify_stp_rapid_pvst_request(commands)
    if global_request:
        requests.append(global_request)
    if interfaces_request:
        requests.append(interfaces_request)
    if mstp_requests:
        requests.append(mstp_requests)
    if pvst_request:
        requests.append(pvst_request)
    if rapid_pvst_request:
        requests.append(rapid_pvst_request)
    return requests