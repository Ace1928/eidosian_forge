from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from urllib.parse import quote
def mk_ts_delete(want_ts, have):
    requests = []
    cur_ts = None
    del_ts = {}
    for cts in have.get('trust_stores') or []:
        if cts.get('name') == want_ts.get('name'):
            cur_ts = cts
            break
    if cur_ts:
        for k, v in want_ts.items():
            if v is not None and k != 'name':
                if v == cur_ts.get(k) or isinstance(v, list):
                    del_ts[k] = v
    if len(del_ts) == 0 and len(want_ts) <= 1:
        requests = [{'path': TRUST_STORE_PATH + '=' + want_ts.get('name'), 'method': DELETE}]
    else:
        for k, v in del_ts.items():
            if isinstance(v, list):
                for li in v:
                    if li in (cur_ts.get(k) or []):
                        requests.append({'path': TRUST_STORE_PATH + '=' + want_ts.get('name') + '/config/' + k.replace('_', '-') + '=' + quote(li, safe=''), 'method': DELETE})
            else:
                requests.append({'path': TRUST_STORE_PATH + '=' + want_ts.get('name') + '/config/' + k.replace('_', '-'), 'method': DELETE})
    return requests