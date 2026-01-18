from __future__ import absolute_import, division, print_function
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible.module_utils.connection import ConnectionError
def get_all_bgp_af_redistribute(module, vrfs, af_redis_params_map):
    """Get all BGP Global Address Family Redistribute configurations available in chassis"""
    all_af_redis_data = []
    ret_redis_data = []
    for vrf_name in vrfs:
        af_redis_data = {}
        request_path = '%s=%s/table-connections' % (network_instance_path, vrf_name)
        request = {'path': request_path, 'method': GET}
        try:
            response = edit_config(module, to_request(module, request))
        except ConnectionError as exc:
            module.fail_json(msg=str(exc), code=exc.code)
        if 'openconfig-network-instance:table-connections' in response[0][1]:
            af_redis_data.update({vrf_name: response[0][1]['openconfig-network-instance:table-connections']})
        if af_redis_data:
            all_af_redis_data.append(af_redis_data)
    if all_af_redis_data:
        for vrf_name in vrfs:
            key = vrf_name
            val = next((af_redis_data for af_redis_data in all_af_redis_data if vrf_name in af_redis_data), None)
            if not val:
                continue
            val = val[vrf_name]
            redis_data = val.get('table-connection', [])
            if not redis_data:
                continue
            filtered_redis_data = []
            for e_cfg in redis_data:
                af_redis_data = get_from_params_map(af_redis_params_map, e_cfg)
                if af_redis_data:
                    filtered_redis_data.append(af_redis_data)
            if filtered_redis_data:
                ret_redis_data.append({key: filtered_redis_data})
    return ret_redis_data