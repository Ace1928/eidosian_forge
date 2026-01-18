from __future__ import absolute_import, division, print_function
from copy import deepcopy
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.argspec.bgp_as_paths.bgp_as_paths import Bgp_as_pathsArgs
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible.module_utils.connection import ConnectionError
def get_as_path_list(self):
    url = 'data/openconfig-routing-policy:routing-policy/defined-sets/openconfig-bgp-policy:bgp-defined-sets/as-path-sets'
    method = 'GET'
    request = [{'path': url, 'method': method}]
    try:
        response = edit_config(self._module, to_request(self._module, request))
    except ConnectionError as exc:
        self._module.fail_json(msg=str(exc), code=exc.code)
    as_path_lists = []
    if 'openconfig-bgp-policy:as-path-sets' in response[0][1]:
        temp = response[0][1].get('openconfig-bgp-policy:as-path-sets', {})
        if 'as-path-set' in temp:
            as_path_lists = temp['as-path-set']
    as_path_list_configs = []
    for as_path in as_path_lists:
        result = dict()
        as_name = as_path['as-path-set-name']
        member_config = as_path['config']
        members = member_config.get('as-path-set-member', [])
        permit_str = member_config.get('openconfig-bgp-policy-ext:action', None)
        result['name'] = as_name
        result['members'] = members
        if permit_str and permit_str == 'PERMIT':
            result['permit'] = True
        else:
            result['permit'] = False
        as_path_list_configs.append(result)
    return as_path_list_configs