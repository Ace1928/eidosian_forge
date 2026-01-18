from __future__ import absolute_import, division, print_function
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible.module_utils.connection import ConnectionError
def get_undefined_bgps(want, have, check_neighbors=None):
    if check_neighbors is None:
        check_neighbors = False
    undefined_resources = []
    if not want:
        return undefined_resources
    if not have:
        have = []
    for want_conf in want:
        undefined = {}
        want_bgp_as = want_conf['bgp_as']
        want_vrf = want_conf['vrf_name']
        have_conf = next((conf for conf in have if want_bgp_as == conf['bgp_as'] and want_vrf == conf['vrf_name']), None)
        if not have_conf:
            undefined['bgp_as'] = want_bgp_as
            undefined['vrf_name'] = want_vrf
            undefined_resources.append(undefined)
        if check_neighbors and have_conf:
            want_neighbors = want_conf.get('neighbors', [])
            have_neighbors = have_conf.get('neighbors', [])
            undefined_neighbors = get_undefined_neighbors(want_neighbors, have_neighbors)
            if undefined_neighbors:
                undefined['bgp_as'] = want_bgp_as
                undefined['vrf_name'] = want_vrf
                undefined['neighbors'] = undefined_neighbors
                undefined_resources.append(undefined)
    return undefined_resources