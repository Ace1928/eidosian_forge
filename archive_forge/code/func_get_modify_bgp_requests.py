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
def get_modify_bgp_requests(self, commands, have):
    requests = []
    if not commands:
        return requests
    for cmd in commands:
        edit_path = '%s=%s/%s' % (self.network_instance_path, cmd['vrf_name'], self.protocol_bgp_path)
        if 'peer_group' in cmd and cmd['peer_group']:
            edit_peer_groups_payload, edit_requests = self.build_bgp_peer_groups_payload(cmd['peer_group'], have, cmd['bgp_as'], cmd['vrf_name'])
            edit_peer_groups_path = edit_path + '/peer-groups'
            if edit_requests:
                requests.extend(edit_requests)
            requests.append({'path': edit_peer_groups_path, 'method': PATCH, 'data': edit_peer_groups_payload})
        if 'neighbors' in cmd and cmd['neighbors']:
            edit_neighbors_payload, edit_requests = self.build_bgp_neighbors_payload(cmd['neighbors'], have, cmd['bgp_as'], cmd['vrf_name'])
            edit_neighbors_path = edit_path + '/neighbors'
            if edit_requests:
                requests.extend(edit_requests)
            requests.append({'path': edit_neighbors_path, 'method': PATCH, 'data': edit_neighbors_payload})
    return requests