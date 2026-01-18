from __future__ import absolute_import, division, print_function
from ansible_collections.community.general.plugins.module_utils.hwc_utils import (
def build_delete_parameters(opts):
    params = dict()
    params['delete_publicip'] = False
    params['delete_volume'] = False
    v = expand_delete_servers(opts, None)
    if not is_empty_value(v):
        params['servers'] = v
    return params