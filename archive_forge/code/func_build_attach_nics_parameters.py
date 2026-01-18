from __future__ import absolute_import, division, print_function
from ansible_collections.community.general.plugins.module_utils.hwc_utils import (
def build_attach_nics_parameters(opts):
    params = dict()
    v = expand_attach_nics_nics(opts, None)
    if not is_empty_value(v):
        params['nics'] = v
    return params