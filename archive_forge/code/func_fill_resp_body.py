from __future__ import absolute_import, division, print_function
from ansible_collections.community.general.plugins.module_utils.hwc_utils import (
def fill_resp_body(body):
    result = dict()
    result['destination'] = body.get('destination')
    result['id'] = body.get('id')
    result['nexthop'] = body.get('nexthop')
    result['type'] = body.get('type')
    result['vpc_id'] = body.get('vpc_id')
    return result