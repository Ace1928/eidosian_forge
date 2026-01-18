from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts import Facts
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import to_request
from ansible.module_utils.connection import ConnectionError
def get_modify_keepalive_request(self, vrf_name, keepalive_interval):
    request = None
    method = PATCH
    payload = {}
    if keepalive_interval is not None:
        payload['keepalive-interval'] = keepalive_interval
    if payload:
        url = '%s=%s/%s/global/%s' % (self.network_instance_path, vrf_name, self.protocol_bgp_path, self.keepalive_path)
        request = {'path': url, 'method': method, 'data': payload}
    return request