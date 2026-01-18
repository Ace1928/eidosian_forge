from __future__ import absolute_import, division, print_function
from ansible_collections.community.general.plugins.module_utils.hwc_utils import (
def build_update_parameters(opts):
    params = dict()
    v = navigate_value(opts, ['dhcp_enable'], None)
    if v is not None:
        params['dhcp_enable'] = v
    v = expand_update_dns_list(opts, None)
    if v is not None:
        params['dnsList'] = v
    v = navigate_value(opts, ['name'], None)
    if not is_empty_value(v):
        params['name'] = v
    v = expand_update_primary_dns(opts, None)
    if v is not None:
        params['primary_dns'] = v
    v = expand_update_secondary_dns(opts, None)
    if v is not None:
        params['secondary_dns'] = v
    if not params:
        return params
    params = {'subnet': params}
    return params