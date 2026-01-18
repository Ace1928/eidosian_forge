from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts import Facts
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import to_request
from ansible.module_utils.connection import ConnectionError
def get_delete_max_med_requests(self, vrf_name, max_med, match):
    requests = []
    match_max_med = match.get('max_med', None)
    if not max_med or not match_max_med:
        return requests
    generic_del_path = '%s=%s/%s/global/' % (self.network_instance_path, vrf_name, self.protocol_bgp_path)
    match_max_med_on_startup = match.get('max_med', {}).get('on_startup')
    if match_max_med_on_startup:
        requests.append({'path': generic_del_path + 'max-med/config/time', 'method': DELETE})
        requests.append({'path': generic_del_path + 'max-med/config/max-med-val', 'method': DELETE})
    return requests