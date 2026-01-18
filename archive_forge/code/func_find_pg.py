from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts import Facts
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.bgp_utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import to_request
from ansible.module_utils.connection import ConnectionError
from copy import deepcopy
def find_pg(self, have, bgp_as, vrf_name, peergroup):
    mat_dict = next((m_peer for m_peer in have if m_peer['bgp_as'] == bgp_as and m_peer['vrf_name'] == vrf_name), None)
    if mat_dict and mat_dict.get('peer_group', None) is not None:
        mat_pg = next((m for m in mat_dict['peer_group'] if m['name'] == peergroup['name']), None)
        return mat_pg