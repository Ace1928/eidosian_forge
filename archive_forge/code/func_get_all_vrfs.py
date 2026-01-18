from __future__ import absolute_import, division, print_function
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible.module_utils.connection import ConnectionError
def get_all_vrfs(module):
    """Get all VRF configurations available in chassis"""
    all_vrfs = []
    ret = []
    request = {'path': 'data/sonic-vrf:sonic-vrf/VRF/VRF_LIST', 'method': GET}
    try:
        response = edit_config(module, to_request(module, request))
    except ConnectionError as exc:
        module.fail_json(msg=str(exc), code=exc.code)
    if 'sonic-vrf:VRF_LIST' in response[0][1]:
        all_vrf_data = response[0][1].get('sonic-vrf:VRF_LIST', [])
        if all_vrf_data:
            for vrf_data in all_vrf_data:
                all_vrfs.append(vrf_data['vrf_name'])
    return all_vrfs