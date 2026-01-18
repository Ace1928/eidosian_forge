from __future__ import absolute_import, division, print_function
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible.module_utils.connection import ConnectionError
def get_bgp_af_data(module, af_params_map):
    vrf_list = get_all_vrfs(module)
    data = get_all_bgp_globals(module, vrf_list)
    objs = []
    for conf in data:
        if conf:
            obj = get_bgp_global_af_data(conf, af_params_map)
            if obj:
                objs.append(obj)
    return objs