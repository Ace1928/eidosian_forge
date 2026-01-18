from __future__ import absolute_import, division, print_function
from ansible_collections.community.general.plugins.module_utils.hwc_utils import (
def expand_create_metadata(d, array_index):
    r = dict()
    v = navigate_value(d, ['encryption_id'], array_index)
    if not is_empty_value(v):
        r['__system__cmkid'] = v
    v = expand_create_metadata_system_encrypted(d, array_index)
    if not is_empty_value(v):
        r['__system__encrypted'] = v
    v = expand_create_metadata_full_clone(d, array_index)
    if not is_empty_value(v):
        r['full_clone'] = v
    v = expand_create_metadata_hw_passthrough(d, array_index)
    if not is_empty_value(v):
        r['hw:passthrough'] = v
    return r