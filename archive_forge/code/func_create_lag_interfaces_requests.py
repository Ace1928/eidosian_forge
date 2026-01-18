from __future__ import absolute_import, division, print_function
import json
from copy import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts import Facts
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.formatted_diff_utils import (
from ansible.module_utils._text import to_native
from ansible.module_utils.connection import ConnectionError
import traceback
def create_lag_interfaces_requests(self, commands):
    requests = []
    for i in commands:
        if i.get('members') and i['members'].get('interfaces'):
            interfaces = i['members']['interfaces']
        else:
            continue
        for each in interfaces:
            edit_payload = self.build_create_payload_member(i['name'])
            template = 'data/openconfig-interfaces:interfaces/interface=%s/openconfig-if-ethernet:ethernet/config/openconfig-if-aggregate:aggregate-id'
            edit_path = template % quote(each['member'], safe='')
            request = {'path': edit_path, 'method': PATCH, 'data': edit_payload}
            requests.append(request)
    return requests