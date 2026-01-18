from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts import Facts
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.formatted_diff_utils import (
from ansible.module_utils.connection import ConnectionError
def get_merge_requests(self, configs, have):
    requests = []
    enable_auth_config = configs.get('enable_ntp_auth', None)
    if enable_auth_config is not None:
        enable_auth_request = self.get_create_enable_ntp_auth_requests(enable_auth_config, have)
        if enable_auth_request:
            requests.extend(enable_auth_request)
    src_intf_config = configs.get('source_interfaces', None)
    if src_intf_config:
        src_intf_request = self.get_create_source_interface_requests(src_intf_config, have)
        if src_intf_request:
            requests.extend(src_intf_request)
    keys_config = configs.get('ntp_keys', None)
    if keys_config:
        keys_request = self.get_create_keys_requests(keys_config, have)
        if keys_request:
            requests.extend(keys_request)
    servers_config = configs.get('servers', None)
    if servers_config:
        servers_request = self.get_create_servers_requests(servers_config, have)
        if servers_request:
            requests.extend(servers_request)
    trusted_key_config = configs.get('trusted_keys', None)
    if trusted_key_config:
        trusted_key_request = self.get_create_trusted_key_requests(trusted_key_config, have)
        if trusted_key_request:
            requests.extend(trusted_key_request)
    vrf_config = configs.get('vrf', None)
    if vrf_config:
        vrf_request = self.get_create_vrf_requests(vrf_config, have)
        if vrf_request:
            requests.extend(vrf_request)
    return requests