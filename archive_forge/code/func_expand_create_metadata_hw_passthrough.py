from __future__ import absolute_import, division, print_function
from ansible_collections.community.general.plugins.module_utils.hwc_utils import (
def expand_create_metadata_hw_passthrough(d, array_index):
    v = navigate_value(d, ['enable_scsi'], array_index)
    if v is None:
        return v
    return 'true' if v else 'false'