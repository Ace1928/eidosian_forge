from __future__ import absolute_import, division, print_function
from ansible_collections.community.general.plugins.module_utils.hwc_utils import (
def flatten_enable_auto_recovery(d, array_index):
    v = navigate_value(d, ['read_auto_recovery', 'support_auto_recovery'], array_index)
    return v == 'true'