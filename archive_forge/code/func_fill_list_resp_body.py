from __future__ import absolute_import, division, print_function
from ansible_collections.community.general.plugins.module_utils.hwc_utils import (
def fill_list_resp_body(body):
    result = dict()
    result['id'] = body.get('id')
    result['ip_address'] = body.get('ip_address')
    result['subnet_id'] = body.get('subnet_id')
    return result