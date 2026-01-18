from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts import Facts
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.formatted_diff_utils import (
from ansible.module_utils.connection import ConnectionError
def get_delete_requests(self, configs, delete_all):
    requests = []
    if delete_all:
        all_ntp_request = self.get_delete_all_ntp_requests(configs)
        if all_ntp_request:
            requests.extend(all_ntp_request)
        return requests
    src_intf_config = configs.get('source_interfaces', None)
    if src_intf_config:
        src_intf_request = self.get_delete_source_interface_requests(src_intf_config)
        if src_intf_request:
            requests.extend(src_intf_request)
    servers_config = configs.get('servers', None)
    if servers_config:
        servers_request = self.get_delete_servers_requests(servers_config)
        if servers_request:
            requests.extend(servers_request)
    trusted_key_config = configs.get('trusted_keys', None)
    if trusted_key_config:
        trusted_key_request = self.get_delete_trusted_key_requests(trusted_key_config)
        if trusted_key_request:
            requests.extend(trusted_key_request)
    keys_config = configs.get('ntp_keys', None)
    if keys_config:
        keys_request = self.get_delete_keys_requests(keys_config)
        if keys_request:
            requests.extend(keys_request)
    enable_auth_config = configs.get('enable_ntp_auth', None)
    if enable_auth_config is not None:
        enable_auth_request = self.get_delete_enable_ntp_auth_requests(enable_auth_config)
        if enable_auth_request:
            requests.extend(enable_auth_request)
    vrf_config = configs.get('vrf', None)
    if vrf_config:
        vrf_request = self.get_delete_vrf_requests(vrf_config)
        if vrf_request:
            requests.extend(vrf_request)
    return requests