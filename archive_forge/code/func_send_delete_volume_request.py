from __future__ import absolute_import, division, print_function
from ansible_collections.community.general.plugins.module_utils.hwc_utils import (
def send_delete_volume_request(module, params, client, info):
    path_parameters = {'volume_id': ['volume_id']}
    data = dict(((key, navigate_value(info, path)) for key, path in path_parameters.items()))
    url = build_path(module, 'cloudservers/{id}/detachvolume/{volume_id}', data)
    try:
        r = client.delete(url, params)
    except HwcClientException as ex:
        msg = 'module(hwc_ecs_instance): error running api(delete_volume), error: %s' % str(ex)
        module.fail_json(msg=msg)
    return r