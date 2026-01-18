from __future__ import absolute_import, division, print_function
from ansible_collections.community.general.plugins.module_utils.hwc_utils import (
def fill_list_resp_attachments(value):
    if not value:
        return None
    result = []
    for item in value:
        val = dict()
        val['attached_at'] = item.get('attached_at')
        val['attachment_id'] = item.get('attachment_id')
        val['device'] = item.get('device')
        val['server_id'] = item.get('server_id')
        result.append(val)
    return result