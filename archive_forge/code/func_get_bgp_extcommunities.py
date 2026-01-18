from __future__ import absolute_import, division, print_function
from copy import deepcopy
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.argspec.bgp_ext_communities.bgp_ext_communities import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible.module_utils.connection import ConnectionError
def get_bgp_extcommunities(self):
    url = 'data/openconfig-routing-policy:routing-policy/defined-sets/openconfig-bgp-policy:bgp-defined-sets/ext-community-sets'
    method = 'GET'
    request = [{'path': url, 'method': method}]
    try:
        response = edit_config(self._module, to_request(self._module, request))
    except ConnectionError as exc:
        self._module.fail_json(msg=str(exc), code=exc.code)
    bgp_extcommunities = []
    if 'openconfig-bgp-policy:ext-community-sets' in response[0][1]:
        temp = response[0][1].get('openconfig-bgp-policy:ext-community-sets', {})
        if 'ext-community-set' in temp:
            bgp_extcommunities = temp['ext-community-set']
    bgp_extcommunities_configs = []
    for bgp_extcommunity in bgp_extcommunities:
        result = dict()
        name = bgp_extcommunity['ext-community-set-name']
        member_config = bgp_extcommunity['config']
        match = member_config['match-set-options']
        permit_str = member_config.get('openconfig-bgp-policy-ext:action', None)
        members = member_config.get('ext-community-member', [])
        result['name'] = str(name)
        result['match'] = match.lower()
        result['members'] = dict()
        result['type'] = 'standard'
        result['permit'] = False
        if permit_str and permit_str == 'PERMIT':
            result['permit'] = True
        if members:
            result['type'] = 'expanded' if 'REGEX' in members[0] else 'standard'
        if result['type'] == 'expanded':
            members = [':'.join(i.split(':')[1:]) for i in members]
            members_list = list(map(str, members))
            members_list.sort()
            result['members'] = {'regex': members_list}
        else:
            rt = list()
            soo = list()
            for member in members:
                if member.startswith('route-origin'):
                    soo.append(':'.join(member.split(':')[1:]))
                else:
                    rt.append(':'.join(member.split(':')[1:]))
            route_target_list = list(map(str, rt))
            route_origin_list = list(map(str, soo))
            route_target_list.sort()
            route_origin_list.sort()
            if route_target_list and len(route_target_list) > 0:
                result['members']['route_target'] = route_target_list
            if route_origin_list and len(route_origin_list) > 0:
                result['members']['route_origin'] = route_origin_list
        bgp_extcommunities_configs.append(result)
    return bgp_extcommunities_configs