from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts import Facts
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.formatted_diff_utils import (
from ansible.module_utils.connection import ConnectionError
def get_delete_all_ntp_requests(self, configs):
    requests = []
    method = DELETE
    src_intf_config = configs.get('source_interfaces', None)
    vrf_config = configs.get('vrf', None)
    enable_auth_config = configs.get('enable_ntp_auth', None)
    trusted_key_config = configs.get('trusted_keys', None)
    if src_intf_config or vrf_config or trusted_key_config or (enable_auth_config is not None):
        url = 'data/openconfig-system:system/ntp'
        request = {'path': url, 'method': method}
        requests.append(request)
    servers_config = configs.get('servers', None)
    if servers_config:
        url = 'data/openconfig-system:system/ntp/servers'
        request = {'path': url, 'method': method}
        requests.append(request)
    keys_config = configs.get('ntp_keys', None)
    if keys_config:
        url = 'data/openconfig-system:system/ntp/ntp-keys'
        request = {'path': url, 'method': method}
        requests.append(request)
    return requests