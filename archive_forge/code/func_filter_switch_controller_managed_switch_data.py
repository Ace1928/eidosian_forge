from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.connection import Connection
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortimanager.common import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.data_post_processor import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.comparison import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.comparison import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.comparison import (
def filter_switch_controller_managed_switch_data(json):
    option_list = ['settings_802_1X', 'access_profile', 'custom_command', 'delayed_restart_trigger', 'description', 'dhcp_server_access_list', 'dhcp_snooping_static_client', 'directly_connected', 'dynamic_capability', 'dynamically_discovered', 'firmware_provision', 'firmware_provision_latest', 'firmware_provision_version', 'flow_identity', 'fsw_wan1_admin', 'fsw_wan1_peer', 'fsw_wan2_admin', 'fsw_wan2_peer', 'igmp_snooping', 'ip_source_guard', 'l3_discovered', 'max_allowed_trunk_members', 'mclag_igmp_snooping_aware', 'mirror', 'name', 'override_snmp_community', 'override_snmp_sysinfo', 'override_snmp_trap_threshold', 'override_snmp_user', 'owner_vdom', 'poe_detection_type', 'poe_lldp_detection', 'poe_pre_standard_detection', 'ports', 'pre_provisioned', 'ptp_profile', 'ptp_status', 'qos_drop_policy', 'qos_red_probability', 'remote_log', 'route_offload', 'route_offload_mclag', 'route_offload_router', 'sn', 'snmp_community', 'snmp_sysinfo', 'snmp_trap_threshold', 'snmp_user', 'staged_image_version', 'static_mac', 'storm_control', 'stp_instance', 'stp_settings', 'switch_device_tag', 'switch_dhcp_opt43_key', 'switch_id', 'switch_log', 'switch_profile', 'switch_stp_settings', 'tdr_supported', 'type', 'version']
    json = remove_invalid_fields(json)
    dictionary = {}
    for attribute in option_list:
        if attribute in json and json[attribute] is not None:
            dictionary[attribute] = json[attribute]
    return dictionary