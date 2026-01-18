from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts import Facts
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import to_request
from ansible.module_utils.connection import ConnectionError
def get_delete_bestpath_requests(self, vrf_name, bestpath, match):
    requests = []
    match_bestpath = match.get('bestpath', None)
    if not bestpath or not match_bestpath:
        return requests
    route_selection_del_path = '%s=%s/%s/global/route-selection-options/config/' % (self.network_instance_path, vrf_name, self.protocol_bgp_path)
    multi_paths_del_path = '%s=%s/%s/global/use-multiple-paths/ebgp/config/' % (self.network_instance_path, vrf_name, self.protocol_bgp_path)
    generic_del_path = '%s=%s/%s/global/' % (self.network_instance_path, vrf_name, self.protocol_bgp_path)
    if bestpath.get('compare_routerid', None) and match_bestpath.get('compare_routerid', None):
        url = '%s=%s/%s/global/route-selection-options' % (self.network_instance_path, vrf_name, self.protocol_bgp_path)
        route_selection_cfg = {}
        route_selection_cfg['external-compare-router-id'] = False
        payload = {'route-selection-options': {'config': route_selection_cfg}}
        requests.append({'path': url, 'data': payload, 'method': PATCH})
    match_as_path = match_bestpath.get('as_path', None)
    as_path = bestpath.get('as_path', None)
    if as_path and match_as_path:
        if as_path.get('confed', None) is not None and match_as_path.get('confed', None):
            requests.append({'path': route_selection_del_path + 'compare-confed-as-path', 'method': DELETE})
        if as_path.get('ignore', None) is not None and match_as_path.get('ignore', None):
            requests.append({'path': route_selection_del_path + 'ignore-as-path-length', 'method': DELETE})
        if as_path.get('multipath_relax', None) is not None and match_as_path.get('multipath_relax', None):
            requests.append({'path': multi_paths_del_path + 'allow-multiple-as', 'method': DELETE})
        if as_path.get('multipath_relax_as_set', None) is not None and match_as_path.get('multipath_relax_as_set', None):
            requests.append({'path': multi_paths_del_path + 'as-set', 'method': DELETE})
    match_med = match_bestpath.get('med', None)
    med = bestpath.get('med', None)
    if med and match_med:
        if med.get('confed', None) is not None and match_med.get('confed', None):
            requests.append({'path': route_selection_del_path + 'med-confed', 'method': DELETE})
        if med.get('missing_as_worst', None) is not None and match_med.get('missing_as_worst', None):
            requests.append({'path': route_selection_del_path + 'med-missing-as-worst', 'method': DELETE})
        if med.get('always_compare_med', None) is not None and match_med.get('always_compare_med', None):
            requests.append({'path': route_selection_del_path + 'always-compare-med', 'method': DELETE})
        if med.get('max_med_val', None) is not None and match_med.get('max_med_val', None):
            requests.append({'path': generic_del_path + 'max-med/config/admin-max-med-val', 'method': DELETE})
    return requests