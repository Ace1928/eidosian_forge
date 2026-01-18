from __future__ import absolute_import, division, print_function
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible.module_utils.connection import ConnectionError
def get_bgp_global_af_data(data, af_params_map):
    ret_af_data = {}
    for key, val in data.items():
        if key == 'global':
            if 'afi-safis' in val and 'afi-safi' in val['afi-safis']:
                global_af_data = []
                raw_af_data = val['afi-safis']['afi-safi']
                for each_af_data in raw_af_data:
                    af_data = get_from_params_map(af_params_map, each_af_data)
                    if af_data:
                        global_af_data.append(af_data)
                ret_af_data.update({'address_family': global_af_data})
            if 'config' in val and 'as' in val['config']:
                as_val = val['config']['as']
                ret_af_data.update({'bgp_as': as_val})
        if key == 'vrf_name':
            ret_af_data.update({'vrf_name': val})
    return ret_af_data