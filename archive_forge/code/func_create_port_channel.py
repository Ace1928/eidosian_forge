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
def create_port_channel(self, cmd):
    requests = []
    path = 'data/openconfig-interfaces:interfaces'
    for i in cmd:
        payload = self.build_create_payload_portchannel(i['name'], i.get('mode', None))
        request = {'path': path, 'method': PATCH, 'data': payload}
        requests.append(request)
    return requests