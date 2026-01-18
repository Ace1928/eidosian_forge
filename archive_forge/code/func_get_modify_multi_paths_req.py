from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts import Facts
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import to_request
from ansible.module_utils.connection import ConnectionError
def get_modify_multi_paths_req(self, vrf_name, as_path):
    request = None
    if not as_path:
        return request
    method = PATCH
    multipath_cfg = {}
    as_path_multipath_relax = as_path.get('multipath_relax', None)
    as_path_multipath_relax_as_set = as_path.get('multipath_relax_as_set', None)
    if as_path_multipath_relax is not None:
        multipath_cfg['allow-multiple-as'] = as_path_multipath_relax
    if as_path_multipath_relax_as_set is not None:
        multipath_cfg['as-set'] = as_path_multipath_relax_as_set
    payload = {'openconfig-network-instance:config': multipath_cfg}
    if payload:
        url = '%s=%s/%s/global/use-multiple-paths/ebgp/config' % (self.network_instance_path, vrf_name, self.protocol_bgp_path)
        request = {'path': url, 'method': method, 'data': payload}
    return request