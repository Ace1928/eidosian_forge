from __future__ import absolute_import, division, print_function
from ansible_collections.community.general.plugins.module_utils.hwc_utils import (
def fill_read_resp_os_extended_volumes_volumes_attached(value):
    if not value:
        return None
    result = []
    for item in value:
        val = dict()
        val['bootIndex'] = item.get('bootIndex')
        val['device'] = item.get('device')
        val['id'] = item.get('id')
        result.append(val)
    return result