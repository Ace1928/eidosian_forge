from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts import Facts
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible.module_utils.connection import ConnectionError
def get_delete_src_ip_request(self, conf, matched, name, del_source_ip):
    requests = []
    url = 'data/sonic-vxlan:sonic-vxlan/VXLAN_TUNNEL/VXLAN_TUNNEL_LIST={name}/src_ip'
    is_change_needed = False
    if matched:
        matched_source_ip = matched.get('source_ip', None)
        if matched_source_ip and matched_source_ip == del_source_ip:
            is_change_needed = True
    if is_change_needed:
        request = {'path': url.format(name=name), 'method': DELETE}
        requests.append(request)
    return requests