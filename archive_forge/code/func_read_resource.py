from __future__ import absolute_import, division, print_function
from ansible_collections.community.general.plugins.module_utils.hwc_utils import (
def read_resource(config, exclude_output=False):
    module = config.module
    client = config.client(get_region(module), 'vpc', 'project')
    res = {}
    r = send_read_request(module, client)
    res['read'] = fill_read_resp_body(r)
    return update_properties(module, res, None, exclude_output)