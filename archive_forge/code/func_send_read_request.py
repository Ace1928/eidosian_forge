from __future__ import absolute_import, division, print_function
from ansible_collections.community.general.plugins.module_utils.hwc_utils import (
def send_read_request(module, client):
    url = build_path(module, 'privateips/{id}')
    r = None
    try:
        r = client.get(url)
    except HwcClientException as ex:
        msg = 'module(hwc_vpc_private_ip): error running api(read), error: %s' % str(ex)
        module.fail_json(msg=msg)
    return navigate_value(r, ['privateip'], None)