from __future__ import absolute_import, division, print_function
from ansible_collections.community.general.plugins.module_utils.hwc_utils import (
def send_create_request(module, params, client):
    url = 'privateips'
    try:
        r = client.post(url, params)
    except HwcClientException as ex:
        msg = 'module(hwc_vpc_private_ip): error running api(create), error: %s' % str(ex)
        module.fail_json(msg=msg)
    return r