from __future__ import absolute_import, division, print_function
from ansible_collections.community.general.plugins.module_utils.hwc_utils import (
def expand_create_metadata_system_encrypted(d, array_index):
    v = navigate_value(d, ['encryption_id'], array_index)
    return '1' if v else ''