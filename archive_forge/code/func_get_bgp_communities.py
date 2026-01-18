from __future__ import absolute_import, division, print_function
from copy import deepcopy
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.argspec.bgp_communities.bgp_communities import Bgp_communitiesArgs
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible.module_utils.connection import ConnectionError
def get_bgp_communities(self):
    url = 'data/openconfig-routing-policy:routing-policy/defined-sets/openconfig-bgp-policy:bgp-defined-sets/community-sets'
    method = 'GET'
    request = [{'path': url, 'method': method}]
    try:
        response = edit_config(self._module, to_request(self._module, request))
    except ConnectionError as exc:
        self._module.fail_json(msg=str(exc), code=exc.code)
    bgp_communities = []
    if 'openconfig-bgp-policy:community-sets' in response[0][1]:
        temp = response[0][1].get('openconfig-bgp-policy:community-sets', {})
        if 'community-set' in temp:
            bgp_communities = temp['community-set']
    bgp_communities_configs = []
    for bgp_community in bgp_communities:
        result = dict()
        name = bgp_community['community-set-name']
        member_config = bgp_community['config']
        match = member_config['match-set-options']
        permit_str = member_config.get('openconfig-bgp-policy-ext:action', None)
        members = member_config.get('community-member', [])
        result['name'] = str(name)
        result['match'] = match
        result['members'] = None
        result['permit'] = False
        if permit_str and permit_str == 'PERMIT':
            result['permit'] = True
        if members:
            result['type'] = 'expanded' if 'REGEX' in members[0] else 'standard'
            if result['type'] == 'expanded':
                members = [':'.join(i.split(':')[1:]) for i in members]
                members.sort()
                result['members'] = {'regex': members}
        else:
            result['type'] = 'standard'
        if result['type'] == 'standard':
            result['local_as'] = None
            result['no_advertise'] = None
            result['no_export'] = None
            result['no_peer'] = None
            for i in members:
                if 'NO_EXPORT_SUBCONFED' in i:
                    result['local_as'] = True
                elif 'NO_ADVERTISE' in i:
                    result['no_advertise'] = True
                elif 'NO_EXPORT' in i:
                    result['no_export'] = True
                elif 'NOPEER' in i:
                    result['no_peer'] = True
        bgp_communities_configs.append(result)
    return bgp_communities_configs