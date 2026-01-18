from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts import Facts
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.formatted_diff_utils import (
def get_delete_all_system_request(self, have):
    requests = []
    if 'hostname' in have:
        request = self.get_hostname_delete_request()
        requests.append(request)
    if 'interface_naming' in have:
        request = self.get_intfname_delete_request()
        requests.append(request)
    if 'anycast_address' in have:
        request = self.get_anycast_delete_request(have['anycast_address'])
        requests.extend(request)
    return requests