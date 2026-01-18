from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts import Facts
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible.module_utils.connection import ConnectionError
def get_delete_single_as_path_member_request(self, name, members):
    url = 'data/openconfig-routing-policy:routing-policy/defined-sets/openconfig-bgp-policy:'
    url = url + 'bgp-defined-sets/as-path-sets/as-path-set={name}/config/{members_param}'
    method = 'DELETE'
    members_params = {'as-path-set-member': ','.join(members)}
    members_str = urlencode(members_params)
    request = {'path': url.format(name=name, members_param=members_str), 'method': method}
    return request