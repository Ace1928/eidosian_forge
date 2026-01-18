from __future__ import absolute_import, division, print_function
from ansible_collections.community.general.plugins.module_utils.hwc_utils import (
def expand_create_request_vpc_info(d, array_index):
    r = dict()
    r['tenant_id'] = ''
    v = navigate_value(d, ['local_vpc_id'], array_index)
    if not is_empty_value(v):
        r['vpc_id'] = v
    return r