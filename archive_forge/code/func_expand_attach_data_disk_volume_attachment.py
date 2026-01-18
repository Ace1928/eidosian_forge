from __future__ import absolute_import, division, print_function
from ansible_collections.community.general.plugins.module_utils.hwc_utils import (
def expand_attach_data_disk_volume_attachment(d, array_index):
    r = dict()
    v = navigate_value(d, ['data_volumes', 'device'], array_index)
    if not is_empty_value(v):
        r['device'] = v
    v = navigate_value(d, ['data_volumes', 'volume_id'], array_index)
    if not is_empty_value(v):
        r['volumeId'] = v
    return r