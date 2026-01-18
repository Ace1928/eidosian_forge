from __future__ import absolute_import, division, print_function
from ansible_collections.community.general.plugins.module_utils.hwc_utils import (
def set_readonly_nics(inputv, curv):
    if not (curv and isinstance(curv, list)):
        return
    if not (inputv and isinstance(inputv, list)):
        return
    lcv = len(curv)
    q = []
    for iv in inputv:
        if len(q) == lcv:
            break
        cv = None
        for j in range(lcv):
            if j in q:
                continue
            cv = curv[j]
            if iv['ip_address'] != cv['ip_address']:
                continue
            q.append(j)
            break
        else:
            continue
        iv['port_id'] = cv.get('port_id')