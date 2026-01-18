from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible.module_utils.connection import ConnectionError
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts import Facts
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.formatted_diff_utils import (
def get_radius_global_payload(self, conf):
    payload = {}
    global_cfg = {}
    if conf.get('auth_type', None):
        global_cfg['auth-type'] = conf['auth_type']
    if conf.get('key', None):
        global_cfg['secret-key'] = conf['key']
    if conf.get('timeout', None):
        global_cfg['timeout'] = conf['timeout']
    if global_cfg:
        payload = {'openconfig-system:config': global_cfg}
    return payload