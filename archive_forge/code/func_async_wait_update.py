from __future__ import absolute_import, division, print_function
from ansible_collections.community.general.plugins.module_utils.hwc_utils import (
def async_wait_update(config, result, client, timeout):
    module = config.module
    path_parameters = {'subnet_id': ['subnet', 'id']}
    data = dict(((key, navigate_value(result, path)) for key, path in path_parameters.items()))
    url = build_path(module, 'subnets/{subnet_id}', data)

    def _query_status():
        r = None
        try:
            r = client.get(url, timeout=timeout)
        except HwcClientException:
            return (None, '')
        try:
            s = navigate_value(r, ['subnet', 'status'])
            return (r, s)
        except Exception:
            return (None, '')
    try:
        return wait_to_finish(['ACTIVE'], ['UNKNOWN'], _query_status, timeout)
    except Exception as ex:
        module.fail_json(msg='module(hwc_vpc_subnet): error waiting for api(update) to be done, error= %s' % str(ex))