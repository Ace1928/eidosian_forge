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
def filter_vpn_ipsec_phase1_data(json):
    option_list = ['acct_verify', 'add_gw_route', 'add_route', 'assign_ip', 'assign_ip_from', 'authmethod', 'authmethod_remote', 'authpasswd', 'authusr', 'authusrgrp', 'auto_negotiate', 'backup_gateway', 'banner', 'cert_id_validation', 'certificate', 'childless_ike', 'client_auto_negotiate', 'client_keep_alive', 'comments', 'dhcp_ra_giaddr', 'dhcp6_ra_linkaddr', 'dhgrp', 'digital_signature_auth', 'distance', 'dns_mode', 'domain', 'dpd', 'dpd_retrycount', 'dpd_retryinterval', 'eap', 'eap_exclude_peergrp', 'eap_identity', 'enforce_unique_id', 'esn', 'fec_base', 'fec_codec', 'fec_egress', 'fec_health_check', 'fec_ingress', 'fec_mapping_profile', 'fec_receive_timeout', 'fec_redundant', 'fec_send_timeout', 'fgsp_sync', 'forticlient_enforcement', 'fragmentation', 'fragmentation_mtu', 'group_authentication', 'group_authentication_secret', 'ha_sync_esp_seqno', 'idle_timeout', 'idle_timeoutinterval', 'ike_version', 'inbound_dscp_copy', 'include_local_lan', 'interface', 'internal_domain_list', 'ip_delay_interval', 'ipv4_dns_server1', 'ipv4_dns_server2', 'ipv4_dns_server3', 'ipv4_end_ip', 'ipv4_exclude_range', 'ipv4_name', 'ipv4_netmask', 'ipv4_split_exclude', 'ipv4_split_include', 'ipv4_start_ip', 'ipv4_wins_server1', 'ipv4_wins_server2', 'ipv6_dns_server1', 'ipv6_dns_server2', 'ipv6_dns_server3', 'ipv6_end_ip', 'ipv6_exclude_range', 'ipv6_name', 'ipv6_prefix', 'ipv6_split_exclude', 'ipv6_split_include', 'ipv6_start_ip', 'keepalive', 'keylife', 'local_gw', 'localid', 'localid_type', 'loopback_asymroute', 'mesh_selector_type', 'mode', 'mode_cfg', 'mode_cfg_allow_client_selector', 'name', 'nattraversal', 'negotiate_timeout', 'network_id', 'network_overlay', 'npu_offload', 'peer', 'peergrp', 'peerid', 'peertype', 'ppk', 'ppk_identity', 'ppk_secret', 'priority', 'proposal', 'psksecret', 'psksecret_remote', 'reauth', 'rekey', 'remote_gw', 'remotegw_ddns', 'rsa_signature_format', 'rsa_signature_hash_override', 'save_password', 'send_cert_chain', 'signature_hash_alg', 'split_include_service', 'suite_b', 'type', 'unity_support', 'usrgrp', 'wizard_type', 'xauthtype']
    json = remove_invalid_fields(json)
    dictionary = {}
    for attribute in option_list:
        if attribute in json and json[attribute] is not None:
            dictionary[attribute] = json[attribute]
    return dictionary