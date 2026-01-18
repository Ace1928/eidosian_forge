from __future__ import absolute_import, division, print_function
from copy import deepcopy
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts \
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils \
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
def match_peer_configured(self, route_map_payload, have):
    """Determine if the "match peer ..." condition is already configured for the
        route map statement corresponding to the incoming route map update request
        specified by the "route_map_payload" input parameter. Return the peer string
       if a "match peer" condition is already configured. Otherwise, return an empty
       string"""
    if not route_map_payload or not have:
        return ''
    conf_map_name = route_map_payload.get('name')
    conf_seq_num = route_map_payload['statements']['statement'][0]['name']
    if not conf_map_name or not conf_seq_num:
        return ''
    cmd_rmap_have = self.get_matching_map(conf_map_name, int(conf_seq_num), have)
    if not cmd_rmap_have or not cmd_rmap_have.get('match') or (not cmd_rmap_have['match'].get('peer')):
        return ''
    peer_dict = cmd_rmap_have['match']['peer']
    if peer_dict.get('interface'):
        peer_str = peer_dict['interface']
    elif peer_dict.get('ip'):
        peer_str = peer_dict['ip']
    elif peer_dict.get('ipv6'):
        peer_str = peer_dict['ipv6']
    else:
        return ''
    return peer_str