from __future__ import absolute_import, division, print_function
from ansible_collections.community.general.plugins.module_utils.hwc_utils import (
def expand_create_nics(d, array_index):
    new_ai = dict()
    if array_index:
        new_ai.update(array_index)
    req = []
    v = navigate_value(d, ['nics'], new_ai)
    if not v:
        return req
    n = len(v)
    for i in range(n):
        new_ai['nics'] = i
        transformed = dict()
        v = navigate_value(d, ['nics', 'ip_address'], new_ai)
        if not is_empty_value(v):
            transformed['ip_address'] = v
        v = navigate_value(d, ['nics', 'subnet_id'], new_ai)
        if not is_empty_value(v):
            transformed['subnet_id'] = v
        if transformed:
            req.append(transformed)
    return req