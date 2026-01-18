from __future__ import absolute_import, division, print_function
from ansible_collections.community.general.plugins.module_utils.hwc_utils import (
def send_read_auto_recovery_request(module, client):
    url = build_path(module, 'cloudservers/{id}/autorecovery')
    r = None
    try:
        r = client.get(url)
    except HwcClientException as ex:
        msg = 'module(hwc_ecs_instance): error running api(read_auto_recovery), error: %s' % str(ex)
        module.fail_json(msg=msg)
    return r