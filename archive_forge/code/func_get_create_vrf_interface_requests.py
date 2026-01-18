from __future__ import absolute_import, division, print_function
from copy import deepcopy
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts import Facts
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.formatted_diff_utils import (
from ansible.module_utils.connection import ConnectionError
def get_create_vrf_interface_requests(self, configs, have):
    requests = []
    if not configs:
        return requests
    method = PATCH
    for conf in configs:
        if conf.get('members', None):
            if conf['members'].get('interfaces', None):
                url = 'data/openconfig-network-instance:network-instances/network-instance={0}/interfaces/interface'.format(conf['name'])
                payload = self.build_create_vrf_interface_payload(conf)
                if payload:
                    request = {'path': url, 'method': method, 'data': payload}
                    requests.append(request)
    return requests