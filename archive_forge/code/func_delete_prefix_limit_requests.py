from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts import Facts
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.bgp_utils import (
from ansible.module_utils.connection import ConnectionError
def delete_prefix_limit_requests(self, conf_prefix_limit, mat_prefix_limit, conf_afi_safi_val, url):
    requests = []
    max_prefixes = conf_prefix_limit.get('max_prefixes', None)
    prevent_teardown = conf_prefix_limit.get('prevent_teardown', None)
    restart_timer = conf_prefix_limit.get('restart_timer', None)
    warning_threshold = conf_prefix_limit.get('warning_threshold', None)
    if max_prefixes:
        self.append_delete_request(requests, max_prefixes, mat_prefix_limit, 'max_prefixes', url, self.max_prefixes_path % conf_afi_safi_val)
    if prevent_teardown:
        self.append_delete_request(requests, prevent_teardown, mat_prefix_limit, 'prevent_teardown', url, self.prv_teardown_path % conf_afi_safi_val)
    if restart_timer:
        self.append_delete_request(requests, restart_timer, mat_prefix_limit, 'restart_timer', url, self.restart_timer_path % conf_afi_safi_val)
    if warning_threshold:
        self.append_delete_request(requests, warning_threshold, mat_prefix_limit, 'warning_threshold', url, self.wrn_threshold_path % conf_afi_safi_val)
    return requests