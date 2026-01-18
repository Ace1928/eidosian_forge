from __future__ import absolute_import, division, print_function
from ansible_collections.community.general.plugins.module_utils.hwc_utils import (
def fill_list_resp_request_vpc_info(value):
    if not value:
        return None
    result = dict()
    result['tenant_id'] = value.get('tenant_id')
    result['vpc_id'] = value.get('vpc_id')
    return result