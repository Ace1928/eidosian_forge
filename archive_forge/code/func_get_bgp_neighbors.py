from __future__ import absolute_import, division, print_function
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible.module_utils.connection import ConnectionError
def get_bgp_neighbors(module, vrf_name):
    neighbors_data = None
    get_path = '%s=%s/%s/neighbors' % (network_instance_path, vrf_name, protocol_bgp_path)
    request = {'path': get_path, 'method': GET}
    try:
        response = edit_config(module, to_request(module, request))
    except ConnectionError as exc:
        module.fail_json(msg=str(exc), code=exc.code)
    resp = response[0][1]
    if 'openconfig-network-instance:neighbors' in resp:
        neighbors_data = resp['openconfig-network-instance:neighbors']
    return neighbors_data