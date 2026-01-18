from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.connection import Connection
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortimanager.common import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.data_post_processor import (
def filter_system_fortiguard_data(json):
    option_list = ['antispam_cache', 'antispam_cache_mpercent', 'antispam_cache_mpermille', 'antispam_cache_ttl', 'antispam_expiration', 'antispam_force_off', 'antispam_license', 'antispam_timeout', 'anycast_sdns_server_ip', 'anycast_sdns_server_port', 'auto_firmware_upgrade', 'auto_firmware_upgrade_day', 'auto_firmware_upgrade_delay', 'auto_firmware_upgrade_end_hour', 'auto_firmware_upgrade_start_hour', 'auto_join_forticloud', 'ddns_server_ip', 'ddns_server_ip6', 'ddns_server_port', 'FDS_license_expiring_days', 'fortiguard_anycast', 'fortiguard_anycast_source', 'interface', 'interface_select_method', 'load_balance_servers', 'outbreak_prevention_cache', 'outbreak_prevention_cache_mpercent', 'outbreak_prevention_cache_mpermille', 'outbreak_prevention_cache_ttl', 'outbreak_prevention_expiration', 'outbreak_prevention_force_off', 'outbreak_prevention_license', 'outbreak_prevention_timeout', 'persistent_connection', 'port', 'protocol', 'proxy_password', 'proxy_server_ip', 'proxy_server_port', 'proxy_username', 'sandbox_inline_scan', 'sandbox_region', 'sdns_options', 'sdns_server_ip', 'sdns_server_port', 'service_account_id', 'source_ip', 'source_ip6', 'update_build_proxy', 'update_dldb', 'update_extdb', 'update_ffdb', 'update_server_location', 'update_uwdb', 'vdom', 'videofilter_expiration', 'videofilter_license', 'webfilter_cache', 'webfilter_cache_ttl', 'webfilter_expiration', 'webfilter_force_off', 'webfilter_license', 'webfilter_timeout']
    json = remove_invalid_fields(json)
    dictionary = {}
    for attribute in option_list:
        if attribute in json and json[attribute] is not None:
            dictionary[attribute] = json[attribute]
    return dictionary