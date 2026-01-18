from __future__ import absolute_import, division, print_function
from copy import deepcopy
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts \
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils \
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
def get_route_map_delete_match_attr(self, command, cmd_rmap_have, requests):
    """Append to the input list of REST API requests the REST APIs needed
        for deletion of all eligible "match" attributes contained in the
        user input command dict specified by the "command" input parameter
        to this function. Modify the contents of the "command" object to
        remove any attributes that are not currently configured. These
        attributes are not "eligible" for deletion and no REST API "request"
        is generated for them."""
    conf_map_name = command['map_name']
    conf_seq_num = command['sequence_num']
    req_seq_num = str(conf_seq_num)
    match_top = command.get('match')
    if not match_top:
        return
    match_keys = match_top.keys()
    cfg_match_top = cmd_rmap_have.get('match')
    if not cfg_match_top:
        command.pop('match')
        return
    cfg_match_keys = cfg_match_top.keys()
    match_both_keys = set(match_keys).intersection(cfg_match_keys)
    match_pop_keys = set(match_keys).difference(match_both_keys)
    for key in match_pop_keys:
        match_top.pop(key)
    if not match_top or not match_both_keys:
        command.pop('match')
        return
    self.get_route_map_delete_match_bgp(command, match_both_keys, cmd_rmap_have, requests)
    if not command.get('match'):
        if 'match' in command:
            command.pop('match')
        return
    generic_match_rest_attr = {'interface': 'match-interface', 'source_vrf': 'openconfig-routing-policy-ext:match-src-network-instance', 'tag': 'match-tag-set/config/openconfig-routing-policy-ext:tag-value', 'source_protocol': 'config/install-protocol-eq'}
    match_delete_req_base = self.route_map_stmt_base_uri.format(conf_map_name, req_seq_num) + 'conditions/'
    for key in generic_match_rest_attr:
        if key in match_both_keys and match_top[key] == cfg_match_top[key]:
            request_uri = match_delete_req_base + generic_match_rest_attr[key]
            request = {'path': request_uri, 'method': DELETE}
            requests.append(request)
        elif key in match_top:
            match_top.pop(key)
            if not match_top:
                command.pop('match')
                return
    peer_str = ''
    if 'peer' in match_both_keys:
        if match_top['peer'].get('interface') and cfg_match_top['peer'].get('interface') and (match_top['peer']['interface'] == cfg_match_top['peer']['interface']):
            peer_str = match_top['peer']['interface']
        elif match_top['peer'].get('ip') and cfg_match_top['peer'].get('ip') and (match_top['peer']['ip'] == cfg_match_top['peer']['ip']):
            peer_str = match_top['peer']['ip']
        elif match_top['peer'].get('ipv6') and cfg_match_top['peer'].get('ipv6') and (match_top['peer']['ipv6'] == cfg_match_top['peer']['ipv6']):
            peer_str = match_top['peer']['ipv6']
        else:
            match_top.pop('peer')
            if not match_top:
                command.pop('match')
                return
        if peer_str:
            request_uri = match_delete_req_base + 'match-neighbor-set/config/openconfig-routing-policy-ext:address={0}'.format(peer_str)
            request = {'path': request_uri, 'method': DELETE}
            requests.append(request)
    elif 'peer' in match_top:
        match_top.pop('peer')
        if not match_top:
            command.pop('match')
            return
    if 'ip' in match_both_keys and match_top['ip'].get('address') and (match_top['ip']['address'] == cfg_match_top['ip'].get('address')):
        request_uri = match_delete_req_base + 'match-prefix-set/config/prefix-set'
        request = {'path': request_uri, 'method': DELETE}
        requests.append(request)
    elif 'ip' in match_top:
        match_top.pop('ip')
        if not match_top:
            command.pop('match')
            return
    if 'ipv6' in match_both_keys and match_top['ipv6'].get('address') and (match_top['ipv6']['address'] == cfg_match_top['ipv6'].get('address')):
        ipv6_attr_name = 'match-prefix-set/config/openconfig-routing-policy-ext:ipv6-prefix-set'
        request_uri = match_delete_req_base + ipv6_attr_name
        request = {'path': request_uri, 'method': DELETE}
        requests.append(request)
    elif 'ipv6' in match_top:
        match_top.pop('ipv6')
        if not match_top:
            command.pop('match')
            return