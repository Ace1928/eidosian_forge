from __future__ import absolute_import, division, print_function
from ansible_collections.community.general.plugins.module_utils.hwc_utils import (
def expand_set_auto_recovery_support_auto_recovery(d, array_index):
    v = navigate_value(d, ['enable_auto_recovery'], None)
    return None if v is None else str(v).lower()