from __future__ import absolute_import, division, print_function
from ansible_collections.community.general.plugins.module_utils.hwc_utils import (
def build_set_auto_recovery_parameters(opts):
    params = dict()
    v = expand_set_auto_recovery_support_auto_recovery(opts, None)
    if v is not None:
        params['support_auto_recovery'] = v
    return params