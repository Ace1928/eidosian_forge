from __future__ import absolute_import, division, print_function
from ansible_collections.community.general.plugins.module_utils.hwc_utils import (
def expand_list_accept_vpc_info(d, array_index):
    r = dict()
    v = navigate_value(d, ['peering_vpc', 'project_id'], array_index)
    r['tenant_id'] = v
    v = navigate_value(d, ['peering_vpc', 'vpc_id'], array_index)
    r['vpc_id'] = v
    for v in r.values():
        if v is not None:
            return r
    return None