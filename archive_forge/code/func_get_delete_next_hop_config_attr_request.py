from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts import Facts
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.formatted_diff_utils import (
def get_delete_next_hop_config_attr_request(self, vrf_name, prefix, index, attr):
    prefix = prefix.replace('/', '%2F')
    url = '%s=%s/%s/static=%s' % (network_instance_path, vrf_name, protocol_static_routes_path, prefix)
    url += '/next-hops/next-hop=%s/config/%s' % (index, attr)
    request = {'path': url, 'method': DELETE}
    return request