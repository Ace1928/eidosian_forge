from __future__ import absolute_import, division, print_function
from ansible_collections.community.general.plugins.module_utils.hwc_utils import (
def flatten_enable_scsi(d, array_index):
    v = navigate_value(d, ['read', 'metadata', 'hw:passthrough'], array_index)
    if v is None:
        return v
    return True if v in ['true', 'True'] else False