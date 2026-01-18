from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts import Facts
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible.module_utils.connection import ConnectionError
def get_create_evpn_request(self, conf):
    url = 'data/sonic-vxlan:sonic-vxlan/EVPN_NVO/EVPN_NVO_LIST'
    payload = self.build_create_evpn_payload(conf)
    request = {'path': url, 'method': PATCH, 'data': payload}
    return request