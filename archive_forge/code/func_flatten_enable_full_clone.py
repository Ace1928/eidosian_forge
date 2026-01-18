from __future__ import absolute_import, division, print_function
from ansible_collections.community.general.plugins.module_utils.hwc_utils import (
def flatten_enable_full_clone(d, array_index):
    v = navigate_value(d, ['read', 'metadata', 'full_clone'], array_index)
    if v is None:
        return v
    return True if v == '0' else False