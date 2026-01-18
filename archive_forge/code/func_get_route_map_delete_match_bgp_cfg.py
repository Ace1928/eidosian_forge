from __future__ import absolute_import, division, print_function
from copy import deepcopy
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts \
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils \
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
def get_route_map_delete_match_bgp_cfg(self, command, match_both_keys, cmd_rmap_have, requests):
    """Append to the input list of REST API requests the REST APIs needed
        for deletion of all eligible "match" attributes defined within the
        BGP match conditions 'config' section of the openconfig routing-policy
        definitions for "policy-definitions" (route maps)."""
    match_top = command['match']
    cfg_match_top = cmd_rmap_have.get('match')
    conf_map_name = command['map_name']
    conf_seq_num = command['sequence_num']
    req_seq_num = str(conf_seq_num)
    bgp_keys = {'metric', 'origin', 'local_preference', 'community', 'ext_comm', 'ip'}
    delete_bgp_keys = bgp_keys.intersection(match_both_keys)
    if not delete_bgp_keys:
        return
    delete_bgp_attrs = []
    bgp_match_delete_req_base = self.route_map_stmt_base_uri.format(conf_map_name, req_seq_num) + 'conditions/openconfig-bgp-policy:bgp-conditions/config/'
    if 'ip' in delete_bgp_keys:
        if not match_top['ip'].get('next_hop') or not cfg_match_top['ip'].get('next_hop'):
            delete_bgp_keys.remove('ip')
            if 'next_hop' in match_top['ip']:
                match_top['ip'].pop('next_hop')
                if not match_top['ip']:
                    match_top.pop('ip')
                    if not match_top:
                        command.pop('match')
                        return
            if not delete_bgp_keys:
                return
        else:
            if match_top['ip']['next_hop'] == cfg_match_top['ip']['next_hop']:
                request_uri = bgp_match_delete_req_base + 'openconfig-bgp-policy-ext:next-hop-set'
                request = {'path': request_uri, 'method': DELETE}
                requests.append(request)
            else:
                match_top['ip'].pop('next_hop')
                if not match_top['ip']:
                    match_top.pop('ip')
                    if not match_top:
                        command.pop('match')
                        return
            delete_bgp_keys.remove('ip')
            if not delete_bgp_keys:
                return
    bgp_rest_attr = {'community': 'community-set', 'ext_comm': 'ext-community-set', 'local_preference': 'local-pref-eq', 'metric': 'med-eq', 'origin': 'origin-eq'}
    for key in delete_bgp_keys:
        if match_top[key] == cfg_match_top[key]:
            bgp_rest_attr_key = bgp_rest_attr[key]
            delete_bgp_attrs.append(bgp_rest_attr_key)
        else:
            match_top.pop(key)
            if not match_top:
                command.pop('match')
                return
    if not delete_bgp_attrs:
        return
    for attr in delete_bgp_attrs:
        request_uri = bgp_match_delete_req_base + attr
        request = {'path': request_uri, 'method': DELETE}
        requests.append(request)