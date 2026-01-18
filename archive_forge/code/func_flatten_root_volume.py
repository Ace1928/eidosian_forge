from __future__ import absolute_import, division, print_function
from ansible_collections.community.general.plugins.module_utils.hwc_utils import (
def flatten_root_volume(d, array_index):
    result = dict()
    v = navigate_value(d, ['read', 'root_volume', 'device'], array_index)
    result['device'] = v
    v = navigate_value(d, ['read', 'root_volume', 'id'], array_index)
    result['volume_id'] = v
    for v in result.values():
        if v is not None:
            return result
    return None