from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from urllib.parse import quote
def mk_sp_delete(want_sp, have):
    requests = []
    cur_sp = None
    del_sp = {}
    for csp in have.get('security_profiles') or []:
        if csp.get('profile_name') == want_sp.get('profile_name'):
            cur_sp = csp
            break
    if cur_sp:
        for k, v in want_sp.items():
            if v is not None and k != 'profile_name':
                if v == cur_sp.get(k) or isinstance(v, list):
                    del_sp[k] = v
    if len(del_sp) == 0 and len(want_sp) <= 1:
        requests = [{'path': SECURITY_PROFILE_PATH + '=' + want_sp.get('profile_name'), 'method': DELETE}]
    else:
        for k, v in del_sp.items():
            if isinstance(v, list):
                for li in v:
                    if li in (cur_sp.get(k) or []):
                        requests.append({'path': SECURITY_PROFILE_PATH + '=' + want_sp.get('profile_name') + '/config/' + k.replace('_', '-') + '=' + quote(li, safe=''), 'method': DELETE})
            else:
                requests.append({'path': SECURITY_PROFILE_PATH + '=' + want_sp.get('profile_name') + '/config/' + k.replace('_', '-'), 'method': DELETE})
    return requests