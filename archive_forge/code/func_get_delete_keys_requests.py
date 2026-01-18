from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts import Facts
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.formatted_diff_utils import (
from ansible.module_utils.connection import ConnectionError
def get_delete_keys_requests(self, configs):
    requests = []
    method = DELETE
    key_configs = []
    for config in configs:
        key_id = config['key_id']
        url = 'data/openconfig-system:system/ntp/ntp-keys/ntp-key={0}'.format(key_id)
        request = {'path': url, 'method': method}
        requests.append(request)
    return requests