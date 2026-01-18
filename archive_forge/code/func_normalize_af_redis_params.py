from __future__ import absolute_import, division, print_function
from copy import deepcopy
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.argspec.bgp_af.bgp_af import Bgp_afArgs
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.bgp_utils import (
def normalize_af_redis_params(self, af):
    norm_af = list()
    for e_af in af:
        temp = e_af.copy()
        for key, val in e_af.items():
            if 'afi' == key or ('protocol' == key and val):
                if ':' in val:
                    temp[key] = val.split(':')[1].lower()
                if '_' in val:
                    temp[key] = val.split('_')[1].lower()
            elif 'route_map' == key and val:
                temp['route_map'] = val[0]
        norm_af.append(temp)
    return norm_af