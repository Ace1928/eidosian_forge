from __future__ import absolute_import, division, print_function
import json
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts import Facts
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.formatted_diff_utils import (
from ansible.module_utils.connection import ConnectionError
def get_single_user_payload(self, name, role, password, update_pass, match):
    user_cfg = {'username': name}
    if not role and match:
        role = match['role']
    if not password and match:
        password = match['password']
    if role:
        user_cfg['role'] = role
    if password:
        clear_pwd, hashed_pwd = self.get_pwd(password)
        user_cfg['password'] = clear_pwd
        user_cfg['password-hashed'] = hashed_pwd
    pay_load = {'openconfig-system:user': [{'username': name, 'config': user_cfg}]}
    return pay_load