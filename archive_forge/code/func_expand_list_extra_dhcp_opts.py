from __future__ import absolute_import, division, print_function
from ansible_collections.community.general.plugins.module_utils.hwc_utils import (
def expand_list_extra_dhcp_opts(d, array_index):
    new_array_index = dict()
    if array_index:
        new_array_index.update(array_index)
    req = []
    v = navigate_value(d, ['extra_dhcp_opts'], new_array_index)
    n = len(v) if v else 1
    for i in range(n):
        new_array_index['extra_dhcp_opts'] = i
        transformed = dict()
        v = navigate_value(d, ['extra_dhcp_opts', 'name'], new_array_index)
        transformed['opt_name'] = v
        v = navigate_value(d, ['extra_dhcp_opts', 'value'], new_array_index)
        transformed['opt_value'] = v
        for v in transformed.values():
            if v is not None:
                req.append(transformed)
                break
    return req if req else None