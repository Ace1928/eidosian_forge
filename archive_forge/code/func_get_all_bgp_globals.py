from __future__ import absolute_import, division, print_function
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible.module_utils.connection import ConnectionError
def get_all_bgp_globals(module, vrfs):
    """Get all BGP configurations available in chassis"""
    all_bgp_globals = []
    for vrf_name in vrfs:
        get_path = '%s=%s/%s/global' % (network_instance_path, vrf_name, protocol_bgp_path)
        request = {'path': get_path, 'method': GET}
        try:
            response = edit_config(module, to_request(module, request))
        except ConnectionError as exc:
            module.fail_json(msg=str(exc), code=exc.code)
        for resp in response:
            if 'openconfig-network-instance:global' in resp[1]:
                bgp_data = {'global': resp[1].get('openconfig-network-instance:global', {})}
                bgp_data.update({'vrf_name': vrf_name})
                all_bgp_globals.append(bgp_data)
    return all_bgp_globals