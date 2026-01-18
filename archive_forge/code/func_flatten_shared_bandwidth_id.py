from __future__ import absolute_import, division, print_function
from ansible_collections.community.general.plugins.module_utils.hwc_utils import (
def flatten_shared_bandwidth_id(d, array_index, current_value, exclude_output):
    v = navigate_value(d, ['read', 'bandwidth_id'], array_index)
    v1 = navigate_value(d, ['read', 'bandwidth_share_type'], array_index)
    return v if v1 and v1 == 'WHOLE' else current_value