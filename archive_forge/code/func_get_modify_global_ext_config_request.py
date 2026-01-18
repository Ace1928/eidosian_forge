from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible.module_utils.connection import ConnectionError
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts import Facts
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.formatted_diff_utils import (
def get_modify_global_ext_config_request(self, conf):
    request = None
    url = 'data/openconfig-system:system/aaa/server-groups/server-group=RADIUS/openconfig-aaa-radius-ext:radius/config'
    payload = self.get_radius_global_ext_payload(conf)
    if payload:
        request = {'path': url, 'method': PATCH, 'data': payload}
    return request