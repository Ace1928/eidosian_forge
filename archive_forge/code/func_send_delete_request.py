from __future__ import absolute_import, division, print_function
from ansible_collections.community.general.plugins.module_utils.hwc_utils import (
def send_delete_request(module, params, client):
    url = build_path(module, 'privateips/{id}')
    try:
        r = client.delete(url, params)
    except HwcClientException as ex:
        msg = 'module(hwc_vpc_private_ip): error running api(delete), error: %s' % str(ex)
        module.fail_json(msg=msg)
    return r