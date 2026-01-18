from __future__ import absolute_import, division, print_function
import traceback
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.formatted_diff_utils import (
from ansible.module_utils._text import to_native
from ansible.module_utils.connection import ConnectionError
def get_create_l2_interface_requests(self, configs):
    """Returns a list of requests to add the switchport
        configurations specified in the config list
        """
    requests = []
    if not configs:
        return requests
    url = 'data/openconfig-interfaces:interfaces/interface={}/{}/openconfig-vlan:switched-vlan/config'
    method = PATCH
    for conf in configs:
        name = conf['name']
        key = intf_key
        if name.startswith('PortChannel'):
            key = port_chnl_key
        payload = self.build_create_payload(conf)
        request = {'path': url.format(name, key), 'method': method, 'data': payload}
        requests.append(request)
    return requests