from __future__ import absolute_import, division, print_function
from ansible_collections.community.general.plugins.module_utils.hwc_utils import (
def build_attach_data_disk_parameters(opts, array_index):
    params = dict()
    v = expand_attach_data_disk_volume_attachment(opts, array_index)
    if not is_empty_value(v):
        params['volumeAttachment'] = v
    return params