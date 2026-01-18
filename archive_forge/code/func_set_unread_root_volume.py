from __future__ import absolute_import, division, print_function
from ansible_collections.community.general.plugins.module_utils.hwc_utils import (
def set_unread_root_volume(inputv, curv):
    if not (inputv and isinstance(inputv, dict)):
        return
    if not (curv and isinstance(curv, dict)):
        return
    curv['size'] = inputv.get('size')
    curv['snapshot_id'] = inputv.get('snapshot_id')
    curv['volume_type'] = inputv.get('volume_type')