from __future__ import absolute_import, division, print_function
from ansible_collections.community.general.plugins.module_utils.hwc_utils import (
def fill_read_resp_allowed_address_pairs(value):
    if not value:
        return None
    result = []
    for item in value:
        val = dict()
        val['ip_address'] = item.get('ip_address')
        val['mac_address'] = item.get('mac_address')
        result.append(val)
    return result