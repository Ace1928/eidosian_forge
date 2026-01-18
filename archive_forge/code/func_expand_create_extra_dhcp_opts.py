from __future__ import absolute_import, division, print_function
from ansible_collections.community.general.plugins.module_utils.hwc_utils import (
def expand_create_extra_dhcp_opts(d, array_index):
    new_array_index = dict()
    if array_index:
        new_array_index.update(array_index)
    req = []
    v = navigate_value(d, ['extra_dhcp_opts'], new_array_index)
    if not v:
        return req
    n = len(v)
    for i in range(n):
        new_array_index['extra_dhcp_opts'] = i
        transformed = dict()
        v = navigate_value(d, ['extra_dhcp_opts', 'name'], new_array_index)
        if not is_empty_value(v):
            transformed['opt_name'] = v
        v = navigate_value(d, ['extra_dhcp_opts', 'value'], new_array_index)
        if not is_empty_value(v):
            transformed['opt_value'] = v
        if transformed:
            req.append(transformed)
    return req