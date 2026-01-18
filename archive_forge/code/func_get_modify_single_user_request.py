from __future__ import absolute_import, division, print_function
import json
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts import Facts
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.formatted_diff_utils import (
from ansible.module_utils.connection import ConnectionError
def get_modify_single_user_request(self, conf, match):
    request = None
    name = conf.get('name', None)
    role = conf.get('role', None)
    password = conf.get('password', None)
    update_pass = conf.get('update_password', None)
    if role or (password and update_pass == 'always'):
        url = 'data/openconfig-system:system/aaa/authentication/users/user=%s' % name
        payload = self.get_single_user_payload(name, role, password, update_pass, match)
        request = {'path': url, 'method': PATCH, 'data': payload}
    return request