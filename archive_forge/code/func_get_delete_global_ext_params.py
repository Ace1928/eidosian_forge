from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible.module_utils.connection import ConnectionError
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts import Facts
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.formatted_diff_utils import (
def get_delete_global_ext_params(self, conf, match):
    requests = []
    url = 'data/openconfig-system:system/aaa/server-groups/server-group=RADIUS/openconfig-aaa-radius-ext:radius/config/'
    if conf.get('nas_ip', None) and match.get('nas_ip', None):
        requests.append({'path': url + 'nas-ip-address', 'method': DELETE})
    if conf.get('retransmit', None) and match.get('retransmit', None):
        requests.append({'path': url + 'retransmit-attempts', 'method': DELETE})
    if conf.get('statistics', None) and match.get('statistics', None):
        requests.append({'path': url + 'statistics', 'method': DELETE})
    return requests