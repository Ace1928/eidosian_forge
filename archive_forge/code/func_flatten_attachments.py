from __future__ import absolute_import, division, print_function
from ansible_collections.community.general.plugins.module_utils.hwc_utils import (
def flatten_attachments(d, array_index):
    v = navigate_value(d, ['read', 'attachments'], array_index)
    if not v:
        return None
    n = len(v)
    result = []
    new_ai = dict()
    if array_index:
        new_ai.update(array_index)
    for i in range(n):
        new_ai['read.attachments'] = i
        val = dict()
        v = navigate_value(d, ['read', 'attachments', 'attached_at'], new_ai)
        val['attached_at'] = v
        v = navigate_value(d, ['read', 'attachments', 'attachment_id'], new_ai)
        val['attachment_id'] = v
        v = navigate_value(d, ['read', 'attachments', 'device'], new_ai)
        val['device'] = v
        v = navigate_value(d, ['read', 'attachments', 'server_id'], new_ai)
        val['server_id'] = v
        for v in val.values():
            if v is not None:
                result.append(val)
                break
    return result if result else None