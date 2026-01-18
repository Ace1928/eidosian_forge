from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts import Facts
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.formatted_diff_utils import (
def get_delete_all_aaa_request(self, have):
    requests = []
    if 'authentication' in have and have['authentication']:
        if 'local' in have['authentication']['data'] or 'group' in have['authentication']['data']:
            request = self.get_authentication_method_delete_request()
            requests.append(request)
        if 'fail_through' in have['authentication']['data']:
            request = self.get_failthrough_delete_request()
            requests.append(request)
    return requests