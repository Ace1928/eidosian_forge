from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.connection import Connection
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortimanager.common import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.data_post_processor import (
def filter_switch_controller_global_data(json):
    option_list = ['allow_multiple_interfaces', 'bounce_quarantined_link', 'custom_command', 'default_virtual_switch_vlan', 'dhcp_option82_circuit_id', 'dhcp_option82_format', 'dhcp_option82_remote_id', 'dhcp_server_access_list', 'dhcp_snoop_client_db_exp', 'dhcp_snoop_client_req', 'dhcp_snoop_db_per_port_learn_limit', 'disable_discovery', 'fips_enforce', 'firmware_provision_on_authorization', 'https_image_push', 'log_mac_limit_violations', 'mac_aging_interval', 'mac_event_logging', 'mac_retention_period', 'mac_violation_timer', 'quarantine_mode', 'sn_dns_resolution', 'update_user_device', 'vlan_all_mode', 'vlan_identity', 'vlan_optimization']
    json = remove_invalid_fields(json)
    dictionary = {}
    for attribute in option_list:
        if attribute in json and json[attribute] is not None:
            dictionary[attribute] = json[attribute]
    return dictionary