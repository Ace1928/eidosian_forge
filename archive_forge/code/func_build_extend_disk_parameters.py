from __future__ import absolute_import, division, print_function
from ansible_collections.community.general.plugins.module_utils.hwc_utils import (
def build_extend_disk_parameters(opts):
    params = dict()
    v = expand_extend_disk_os_extend(opts, None)
    if not is_empty_value(v):
        params['os-extend'] = v
    return params