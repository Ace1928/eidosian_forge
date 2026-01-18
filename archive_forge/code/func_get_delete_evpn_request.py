from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts import Facts
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible.module_utils.connection import ConnectionError
def get_delete_evpn_request(self, conf, matched, del_evpn_nvo):
    requests = []
    url = 'data/sonic-vxlan:sonic-vxlan/EVPN_NVO/EVPN_NVO_LIST={evpn_nvo}'
    is_change_needed = False
    if matched:
        matched_evpn_nvo = matched.get('evpn_nvo', None)
        if matched_evpn_nvo and matched_evpn_nvo == del_evpn_nvo:
            is_change_needed = True
    if is_change_needed:
        request = {'path': url.format(evpn_nvo=conf['evpn_nvo']), 'method': DELETE}
        requests.append(request)
    return requests