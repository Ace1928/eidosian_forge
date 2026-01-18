from __future__ import absolute_import, division, print_function
from ansible_collections.community.general.plugins.module_utils.hwc_utils import (
def expand_list_allowed_address_pairs(d, array_index):
    new_array_index = dict()
    if array_index:
        new_array_index.update(array_index)
    req = []
    v = navigate_value(d, ['allowed_address_pairs'], new_array_index)
    n = len(v) if v else 1
    for i in range(n):
        new_array_index['allowed_address_pairs'] = i
        transformed = dict()
        v = navigate_value(d, ['allowed_address_pairs', 'ip_address'], new_array_index)
        transformed['ip_address'] = v
        v = navigate_value(d, ['allowed_address_pairs', 'mac_address'], new_array_index)
        transformed['mac_address'] = v
        for v in transformed.values():
            if v is not None:
                req.append(transformed)
                break
    return req if req else None