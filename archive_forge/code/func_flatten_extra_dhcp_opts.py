from __future__ import absolute_import, division, print_function
from ansible_collections.community.general.plugins.module_utils.hwc_utils import (
def flatten_extra_dhcp_opts(d, array_index, current_value, exclude_output):
    n = 0
    result = current_value
    has_init_value = True
    if result:
        n = len(result)
    else:
        has_init_value = False
        result = []
        v = navigate_value(d, ['read', 'extra_dhcp_opts'], array_index)
        if not v:
            return current_value
        n = len(v)
    new_array_index = dict()
    if array_index:
        new_array_index.update(array_index)
    for i in range(n):
        new_array_index['read.extra_dhcp_opts'] = i
        val = dict()
        if len(result) >= i + 1 and result[i]:
            val = result[i]
        v = navigate_value(d, ['read', 'extra_dhcp_opts', 'opt_name'], new_array_index)
        val['name'] = v
        v = navigate_value(d, ['read', 'extra_dhcp_opts', 'opt_value'], new_array_index)
        val['value'] = v
        if len(result) >= i + 1:
            result[i] = val
        else:
            for v in val.values():
                if v is not None:
                    result.append(val)
                    break
    return result if has_init_value or result else current_value