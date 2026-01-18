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
def get_delete_vrf_specific_peergroup_request(self, vrf_name, peergroup_name):
    requests = []
    delete_neighbor_path = '%s=%s/%s/peer-groups/peer-group=%s' % (self.network_instance_path, vrf_name, self.protocol_bgp_path, peergroup_name)
    return {'path': delete_neighbor_path, 'method': DELETE}