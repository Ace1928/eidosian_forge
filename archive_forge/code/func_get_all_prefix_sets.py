from __future__ import (absolute_import, division, print_function)
from copy import deepcopy
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils \
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.argspec.prefix_lists.prefix_lists import Prefix_listsArgs
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
def get_all_prefix_sets(self):
    """Execute a REST "GET" API to fetch all of the current prefix list configuration
        from the target device."""
    pfx_fetch_spec = 'openconfig-routing-policy:routing-policy/defined-sets/prefix-sets'
    pfx_resp_key = 'openconfig-routing-policy:prefix-sets'
    pfx_set_key = 'prefix-set'
    url = 'data/%s' % pfx_fetch_spec
    method = 'GET'
    request = [{'path': url, 'method': method}]
    try:
        response = edit_config(self._module, to_request(self._module, request))
    except ConnectionError as exc:
        self._module.fail_json(msg=str(exc))
    prefix_lists_unparsed = []
    resp_prefix_set = response[0][1].get(pfx_resp_key, None)
    if resp_prefix_set:
        prefix_lists_unparsed = resp_prefix_set.get(pfx_set_key, None)
    return prefix_lists_unparsed