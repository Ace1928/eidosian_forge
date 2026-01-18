from __future__ import absolute_import, division, print_function
from ansible_collections.community.general.plugins.module_utils.hwc_utils import (
def send_extend_disk_request(module, params, client):
    url = build_path(module, 'cloudvolumes/{id}/action')
    try:
        r = client.post(url, params)
    except HwcClientException as ex:
        msg = 'module(hwc_evs_disk): error running api(extend_disk), error: %s' % str(ex)
        module.fail_json(msg=msg)
    return r