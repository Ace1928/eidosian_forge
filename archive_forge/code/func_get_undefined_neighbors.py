from __future__ import absolute_import, division, print_function
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible.module_utils.connection import ConnectionError
def get_undefined_neighbors(want, have):
    undefined_neighbors = []
    if not want:
        return undefined_neighbors
    if not have:
        have = []
    for want_neighbor in want:
        want_neighbor_val = want_neighbor['neighbor']
        have_neighbor = next((conf for conf in have if want_neighbor_val == conf['neighbor']), None)
        if not have_neighbor:
            undefined_neighbors.append({'neighbor': want_neighbor_val})
    return undefined_neighbors